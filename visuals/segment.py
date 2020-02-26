from panda3d.core import *
import math

class Segment():
    def __init__(self, nb_edges=4, pos=(0, 0, 0), length=15, ratio=0.2, size = 1, color=(0.8, 0.0, 0.3, 1.0)):
        self._EDGES = nb_edges
        self.pos = LVecBase3f(pos)
        self.length = length
        self.color = color
        self.size = size
        self.ratio = ratio
        base.render_head_wireframe = False
    
    def draw(self):
        format = GeomVertexFormat.getV3n3cpt2()
        vdata = GeomVertexData("square", format, Geom.UHDynamic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        color = GeomVertexWriter(vdata, "color")
        circle = Geom(vdata)
        # Create vertices
        vertex.addData3f(self.pos)
        color.addData4f(self.color)

        for v in range(self._EDGES):
            x = self.pos.getX() + (self.size * math.cos((2 * math.pi / self._EDGES) * v))
            y = self.pos.getY() + (self.size * math.sin((2 * math.pi / self._EDGES) * v))
            z = self.pos.getZ() + self.ratio*self.length
            vertex.addData3f(x, y, z)
            color.addData4f(self.color)
        
        #add the peaks of the polygone
        vertex.addData3f(self.pos.getX(), self.pos.getY(), self.pos.getZ())
        color.addData4f(self.color)

        vertex.addData3f(self.pos.getX(), self.pos.getY(), self.pos.getZ()+self.length)
        color.addData4f(self.color)

        # Create triangles
        for t in range(self._EDGES):
            
            #generate first pyramid
            tri = GeomTriangles(Geom.UHDynamic)
            tri.addVertex(self._EDGES+1)
            tri.addVertex(t + 1)
            if (t + 2) > self._EDGES:
                tri.addVertex(1)
            else:
                tri.addVertex(t + 2)
            tri.closePrimitive()
            circle.addPrimitive(tri)

            #on the other side
            tri = GeomTriangles(Geom.UHDynamic)
            
            if (t + 2) > self._EDGES:
                tri.addVertex(1)
            else:
                tri.addVertex(t + 2)
            tri.addVertex(t + 1)

            tri.addVertex(self._EDGES+1)

            tri.closePrimitive()
            circle.addPrimitive(tri)

            #generate second pyramid
            tri = GeomTriangles(Geom.UHDynamic)
            tri.addVertex(self._EDGES+2)
            tri.addVertex(t + 1)
            if (t + 2) > self._EDGES:
                tri.addVertex(1)
            else:
                tri.addVertex(t + 2)
            tri.closePrimitive()
            circle.addPrimitive(tri)

            #on the other side
            tri = GeomTriangles(Geom.UHDynamic)
            
            if (t + 2) > self._EDGES:
                tri.addVertex(1)
            else:
                tri.addVertex(t + 2)
            tri.addVertex(t + 1)

            tri.addVertex(self._EDGES+2)

            tri.closePrimitive()
            circle.addPrimitive(tri)


        gn = GeomNode("Circle")
        gn.addGeom(circle)
        node_path = NodePath(gn)
        node_path.setHpr(0, 0, 0)
        return node_path

def Create3DPoint(name = 'node', position = (0,0,0), color = (0.8, 0.0, 0.3, 1.0), size = 10, parentnode = None):
    array = GeomVertexArrayFormat()
    array.addColumn("vertex", 3, Geom.NTFloat32, Geom.CPoint)
    format = GeomVertexFormat()
    format.addArray(array)
    format = GeomVertexFormat.registerFormat(format)
    vdata = GeomVertexData('point', format, Geom.UHStatic)
    vdata.setNumRows(1)
    vertex = GeomVertexWriter(vdata, 'vertex')

    vertex.addData3f(0.0, 0.0, 0.0)

    points = GeomPoints(Geom.UHStatic) 
    points.addVertex(0)

    geom = Geom(vdata)
    geom.addPrimitive(points)
    pointnode = GeomNode(name)
    pointnode.addGeom(geom)

    pointnp = render.attachNewNode(pointnode)
    pointnp.setPos(position[0], position[1], position[2])
    pointnp.setColor(color[0], color[1], color[2], color[3])
    pointnp.setRenderMode(RenderModeAttrib.M_point, size)
    pointnp.setDepthTest(False)  
    pointnp.setDepthWrite(False)  
    pointnp.setBin("fixed", 40)
    
    if parentnode != None:
        pointnp.reparentTo(parentnode)
    
    
    return pointnp