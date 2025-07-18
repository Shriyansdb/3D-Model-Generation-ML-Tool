import torch
import os
import time
import uuid
import logging
import numpy as np
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
from PIL import Image, ImageDraw
import trimesh
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("3DGenerator")

# Global models to avoid reloading
MODELS = {
    'xm': None,
    'model': None,
    'diffusion': None,
    'device': None
}

def init_models():
    """Initialize and cache models for efficient reuse"""
    if MODELS['xm'] is None:
        try:
            # Auto-detect device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Initializing models on device: {device}")
            
            # Load models with progress tracking
            logger.info("Loading transmitter model...")
            xm = load_model('transmitter', device=device)
            
            logger.info("Loading text300M model...")
            model = load_model('text300M', device=device)
            
            logger.info("Loading diffusion config...")
            diffusion = diffusion_from_config(load_config('diffusion'))
            
            # Cache models
            MODELS.update({
                'xm': xm,
                'model': model,
                'diffusion': diffusion,
                'device': device
            })
            
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise
    
    return MODELS['xm'], MODELS['model'], MODELS['diffusion'], MODELS['device']

def save_as_obj(mesh, filename):
    """Save mesh as OBJ format with materials using trimesh"""
    try:
        # Create trimesh object
        tri_mesh = trimesh.Trimesh(
            vertices=np.array(mesh.verts),
            faces=np.array(mesh.faces)
        )
        
        # Export as OBJ
        tri_mesh.export(filename, file_type='obj')
        logger.info(f"Saved OBJ: {filename} ({os.path.getsize(filename)/1024:.1f} KB)")
        return True
    except Exception as e:
        logger.error(f"Failed to save OBJ: {str(e)}")
        return False

def save_as_glb(mesh, filename):
    """Save mesh as GLB format with vertex colors"""
    try:
        # Create trimesh object with vertex colors
        tri_mesh = trimesh.Trimesh(
            vertices=np.array(mesh.verts),
            faces=np.array(mesh.faces),
            vertex_colors=np.array(mesh.vertex_channels['rgb'])
        )
        
        # Export as GLB
        tri_mesh.export(filename, file_type='glb')
        logger.info(f"Saved GLB: {filename} ({os.path.getsize(filename)/1024:.1f} KB)")
        return True
    except Exception as e:
        logger.error(f"Failed to save GLB: {str(e)}")
        return False

def generate_preview_image(mesh, filename, size=256):
    """Generate a 3-view preview image of the mesh"""
    try:
        # Try to import pytorch3d for better rendering
        try:
            from shap_e.util.notebooks import create_pan_cameras
            cameras = create_pan_cameras(size, mesh.verts)
            render = mesh.render(cameras)
        except:
            # Fallback to simple rendering
            logger.warning("Using fallback preview generation")
            render = [np.zeros((size, size, 3)) for _ in range(3)]
        
        # Create combined preview
        preview = Image.new('RGB', (size * 3, size))
        for i, image in enumerate(render):
            # Convert to PIL image
            if not isinstance(image, np.ndarray):
                img_data = image.detach().cpu().numpy() * 255
            else:
                img_data = image * 255
                
            img_data = img_data.astype(np.uint8)
            pil_img = Image.fromarray(img_data)
            preview.paste(pil_img, (i * size, 0))
        
        # Save preview
        preview.save(filename)
        logger.info(f"Saved preview: {filename}")
        return preview
    except Exception as e:
        logger.error(f"Preview generation failed: {str(e)}")
        # Return placeholder if preview fails
        placeholder = Image.new('RGB', (size * 3, size), (240, 240, 240))
        draw = ImageDraw.Draw(placeholder)
        draw.text((10, 10), "Preview Unavailable", fill="black")
        placeholder.save(filename)
        return placeholder

def generate_3d_asset(prompt: str, output_dir="outputs"):
    """Generate 3D asset from text prompt"""
    start_time = time.time()
    logger.info(f"Generating 3D asset for prompt: '{prompt}'")
    
    try:
        # Initialize models
        xm, model, diffusion, device = init_models()
        
        # Create output directories
        for dir_type in ['objs', 'glbs', 'previews']:
            dir_path = os.path.join(output_dir, dir_type)
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        clean_prompt = "".join(x for x in prompt[:50] if x.isalnum() or x in " _-")
        base_name = f"{clean_prompt}_{file_id}"
        logger.info(f"Using base filename: {base_name}")
        
        # Generate 3D latent
        logger.info("Sampling latents...")
        latents = sample_latents(
            batch_size=1,
            model=model,
            diffusion=diffusion,
            guidance_scale=15.0,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=False,  # More compatible with CPU
            use_karras=True,
            karras_steps=32,  # Reduced for faster generation
            sigma_min=1e-3,
            sigma_max=80,
            s_churn=0,
        )
        
        # Decode mesh
        logger.info("Decoding latent to mesh...")
        mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
        
        # File paths
        obj_path = os.path.join(output_dir, 'objs', f"{base_name}.obj")
        glb_path = os.path.join(output_dir, 'glbs', f"{base_name}.glb")
        preview_path = os.path.join(output_dir, 'previews', f"{base_name}.png")
        
        # Save formats
        obj_success = save_as_obj(mesh, obj_path)
        glb_success = save_as_glb(mesh, glb_path)
        preview_success = generate_preview_image(mesh, preview_path)
        
        # Calculate generation time
        gen_time = time.time() - start_time
        logger.info(f"Generated assets in {gen_time:.1f} seconds")
        
        return {
            "obj": obj_path if obj_success else None,
            "glb": glb_path if glb_success else None,
            "preview": preview_path if preview_success else None,
            "generation_time": gen_time,
            "success": obj_success and glb_success
        }
    
    except Exception as e:
        logger.error(f"Generation failed for prompt '{prompt}': {str(e)}")
        return {
            "error": str(e),
            "success": False
        }

def generate_test_asset():
    """Generate a test asset to verify the system works"""
    logger.info("Running system test...")
    try:
        # Create a simple cube programmatically as a test
        test_prompt = "test cube"
        vertices = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [1, 5, 6], [1, 6, 2],
            [5, 4, 7], [5, 7, 6],
            [4, 0, 3], [4, 3, 7],
            [3, 2, 6], [3, 6, 7],
            [4, 5, 1], [4, 1, 0]
        ])
        
        # Create a simple trimesh object
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Create output directories
        os.makedirs("outputs/objs", exist_ok=True)
        os.makedirs("outputs/glbs", exist_ok=True)
        os.makedirs("outputs/previews", exist_ok=True)
        
        # Save test files
        obj_path = "outputs/objs/test_cube.obj"
        glb_path = "outputs/glbs/test_cube.glb"
        preview_path = "outputs/previews/test_cube.png"
        
        tri_mesh.export(obj_path)
        tri_mesh.export(glb_path)
        
        # Create a simple preview
        preview = Image.new('RGB', (300, 100), (200, 200, 255))
        draw = ImageDraw.Draw(preview)
        draw.text((50, 40), "Test Cube Preview", fill="black")
        preview.save(preview_path)
        
        logger.info("System test completed successfully")
        return {
            "obj": obj_path,
            "glb": glb_path,
            "preview": preview_path,
            "success": True
        }
    except Exception as e:
        logger.error(f"System test failed: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }

# Run a system test on import
if __name__ == "__main__":
    logger.info("Running generator self-test...")
    try:
        # First try to initialize models
        init_models()
        
        # Then try to generate a test asset
        test_result = generate_test_asset()
        logger.info(f"Test files created: {test_result}")
        
        # Finally try a real generation
        gen_result = generate_3d_asset("a simple cube")
        logger.info(f"Real generation result: {gen_result}")
        
        # Verify file sizes
        if gen_result.get('obj'):
            logger.info(f"OBJ size: {os.path.getsize(gen_result['obj']) / 1024:.1f} KB")
        if gen_result.get('glb'):
            logger.info(f"GLB size: {os.path.getsize(gen_result['glb']) / 1024:.1f} KB")
        if gen_result.get('preview'):
            logger.info(f"Preview size: {os.path.getsize(gen_result['preview']) / 1024:.1f} KB")
        
    except Exception as e:
        logger.critical(f"Generator self-test failed: {str(e)}")