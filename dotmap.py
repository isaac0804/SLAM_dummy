import numpy as np
import OpenGL.GL as gl
import pangolin
from multiprocessing import Process, Queue
import g2o

class Point(object):
    def __init__(self, mapp, loc):
        self.pt = loc
        self.frames = []
        self.idxs = []
        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)
# https://github.com/uoip/g2opy/issues/15

class Map(object):
    def __init__(self):
        self.points = []
        self.frames = []
        self.state = None
        self.create_viewer()

    '''
    # *** optimizer ***
    def optimize(self):
        # create a g2o optimizer
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolver(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)
        
        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

        # add frames to graph
        for f in self.frames:
            sbacam = g2o.SBACAM(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3]))
            sbacam.set_cam(f.K[0][0], f.K[1][1], f.K[2][0], f.K[2][1], 1.0)
            
            v_se3 = g2o.VertexCam()
            v_se3.set_id(f.id)
            v_se3.set_estimate(sbacam)
            v_se3.set_fixed(f.id == 0)
            opt.add_vertex(v_se3)
        
        # add points to frames 
        for p in self.points:
            pt = g2o.VertexSBAPointXYZ()
            pt.set_id(p.id + 0x10000)
            pt.set_estimate(p.pt[0:3])
            pt.set_marginalized(True)
            pt.set_fixed(False)
            opt.add_vertex(pt)
        
        opt.set_verbose(True)
        opt.initialize_optimization()
        opt.optimize(20)
    
    
    
    '''

    def create_viewer(self):
        self.q = Queue()
        self.p = Process(target=self.viewer_thread, args=(self.q,))
        self.p.daemon = True
        self.p.start()

    def viewer_thread(self, q):
        self.viewer_init(1080, 720)
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('Main', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 1000),
            # pangolin.ProjectionMatrix(w, h, 230, 230, w//2, h//2, 0.2, 5000),
            pangolin.ModelViewLookAt(0, -10, -8,
                                     0, 0, 0,
                                     0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
        self.dcam.SetHandler(self.handler)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.dcam.Activate(self.scam)

        gl.glPointSize(5)
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawCameras(self.state[0])

        gl.glPointSize(1)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(self.state[1])

        pangolin.FinishFrame()

    def display(self):
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((np.array(poses), np.array(pts)))

