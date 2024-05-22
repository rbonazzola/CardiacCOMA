import pyvista as pv
import numpy as np
from PIL import Image
import imageio
from typing import List, Dict, Union, Literal

def render_mesh_as_png(mesh3D, faces, filename, camera_position='xy', show_edges=False, **kwargs):

	'''  
	Produces a png file representing a static 3D mesh.
	- params
	::mesh3D:: a sequence of Trimesh mesh objects. 
	::faces:: array of F x 3 containing the indices of the mesh's triangular faces.
	::filename:: the name of the output png file. 
	::camera_position:: camera position for pyvista plotter (check relevant docs)

	- return:  
	None, only produces the png file. 
	'''

	pv.set_plot_theme("document")
	plotter = pv.Plotter(off_screen=True, notebook=False)
	connectivity = np.c_[np.ones(faces.shape[0]) * 3, faces].astype(int)

	try:
	    # if mesh3D is torch.Tensor, this your should run OK
            # I am casting to np.single in case I train with 16 bit precision (because VTK doesn't like 16 bit precision)
	    mesh3D = mesh3D.cpu().numpy().astype(np.single)
	except:
	    pass

	mesh = pv.PolyData(mesh3D, connectivity)
	actor = plotter.add_mesh(mesh, show_edges=show_edges)
	plotter.camera_position = camera_position
	plotter.screenshot(filename if filename.endswith("png") else filename + ".png")

    
def merge_pngs(
    images: Union[List[str], List[Image.Image]], 
    output_png: Union[None, str],
    how: Literal['horizontally', 'vertically'] = "horizontally") -> Image.Image:

    '''
    params:
      - pngs: list of either 1) png paths or 2) PIL.Image.Image objects
      - output_png (option): path of the output path
      - how: along which axis to merge images horizontally or vertically
    return:
      - a pillow image with the merged images.
    '''
    
    # Reference:
    # https://www.tutorialspoint.com/python_pillow/Python_pillow_merging_images.htm
    
    # Read images
    if isinstance(images[0], str):
        images = [Image.open(image) for image in images]    
    elif isinstance(images[0], Image.Image):
        pass
    else:
        raise TypeError("'images' argument must be either a list of paths or a list of pillow images.")
    
    x_sizes = [image.size[0] for image in images]
    y_sizes = [image.size[1] for image in images]
    
    if how == "vertically":      
        y_size = sum(y_sizes)  
        x_size = images[0].size[0]      
        y_sizes.insert(0, 0)      
        y_positions = np.cumsum(y_sizes[:-1])    
        positions = [(0, y_position) for y_position in y_positions]
    
    elif how == "horizontally":
        x_size = sum(x_sizes)      
        y_size = images[0].size[1]      
        x_sizes.insert(0, 0)      
        x_positions = np.cumsum(x_sizes[:-1])    
        positions = [(x_position, 0) for x_position in x_positions]
    
    
    new_image = Image.new(
        mode='RGB',
        size=(x_size, y_size),
        color=(250, 250, 250)
    )
    
    for i, image in enumerate(images):        
        new_image.paste(image, positions[i])
        
    if output_png is not None:
        new_image.save(output_png, "PNG")

    return new_image
