import os
import trimesh
import numpy as np
from PIL import Image
from shap_e.util.notebooks import create_pan_cameras

def save_as_obj(mesh, filename):
    with open(filename, 'w') as f:
        mesh.write_obj(f, include_material=True)
    print(f"Saved OBJ: {filename}")

def save_as_glb(mesh, filename):
    # Convert to trimesh object with vertex colors
    tri_mesh = trimesh.Trimesh(
        vertices=mesh.verts,
        faces=mesh.faces,
        vertex_colors=np.array(mesh.vertex_channels['rgb'])
    )
    tri_mesh.export(filename, file_type='glb')
    print(f"Saved GLB: {filename}")

def generate_preview_image(mesh, filename, size=256):
    # Create 3 camera views (front, side, top)
    cameras = create_pan_cameras(size, mesh.verts)
    render = mesh.render(cameras)
    
    # Create combined preview
    preview = Image.new('RGB', (size * 3, size))
    for i, image in enumerate(render):
        pil_img = Image.fromarray((image * 255).astype(np.uint8))
        preview.paste(pil_img, (i * size, 0))
    
    preview.save(filename)
    return preview