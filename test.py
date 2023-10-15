from shapely import Polygon, MultiPolygon
from planestress.pre.geometry import Geometry
from planestress.pre.material import Material


shell = [(0,0), (100, 0), (100, 100), (0, 100)]
hole1 = [[(25, 25), (75, 25), (75, 75), (25, 75)]]
poly1 = Polygon(shell, hole1)
material1 = Material(colour="grey")
geom = Geometry(poly1, material1)
geom.create_mesh(20)
geom.plot_mesh(True, alpha=0.2, materials=True)