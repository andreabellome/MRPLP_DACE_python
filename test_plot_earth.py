import pyvista
from pyvista import examples

light = pyvista.Light()
light.set_direction_angle(30, -20)

earth = examples.planets.load_earth(radius=6378.1)
earth_texture = examples.load_globe_texture()

earth.translate((-30000.0, 0.0, 0.0))

pl = pyvista.Plotter(lighting="none")

cubemap = examples.download_cubemap_space_16k()

_ = pl.add_actor(cubemap.to_skybox())

pl.set_environment_texture(cubemap, True)

pl.add_light(light)
pl.add_mesh(earth, texture=earth_texture, smooth_shading=True)

pl.show()

st = 1
