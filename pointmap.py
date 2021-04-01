import numpy as np
import OpenGL.GL as gl
import pangolin
from frame import poseRt
from multiprocessing import Process, Queue
import g2o
import time

LOCAL_WINDOW = 20

class Point(object):
    def __init__(self, mapp, loc, color):
        self.pt = loc
        self.frames = []
        self.idxs = []
        self.color = np.copy(color)
        self.id = mapp.max_point
        mapp.max_point += 1
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)

    def delete(self):
        for f in self.frames:
            f.pts[f.pts.index(self)] = None
        del self


class Map(object):
    def __init__(self):
        self.points = []
        self.frames = []
        self.max_point = 0
        self.state = None
        self.q = None
        self.create_viewer()

    # *** optimizer ***
    def optimize(self):
        # create a g2o optimizer
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)
        
        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

        if LOCAL_WINDOW is None:
            local_frames = self.frames
        else:
            local_frames = self.frames[-LOCAL_WINDOW:]

        # add frames to graph
        for f in self.frames:
            pose = f.pose
            sbacam = g2o.SBACam(g2o.SE3Quat(pose[0:3,0:3], pose[0:3, 3]))
            sbacam.set_cam(f.k[0][0], f.k[1][1], f.k[0][2], f.k[1][2], 1.0)

            v_se3 = g2o.VertexCam()
            v_se3.set_id(f.id)
            v_se3.set_estimate(sbacam)
            v_se3.set_fixed(f.id <= 1 or f not in local_frames)
            opt.add_vertex(v_se3)
        
        # add points to frames
        PT_ID_OFFSET = 0x10000
        for p in self.points:
            if not any([f in local_frames for f in p.frames]):
                continue
            pt = g2o.VertexSBAPointXYZ()
            pt.set_id(p.id + PT_ID_OFFSET)
            pt.set_estimate(p.pt[0:3])
            pt.set_marginalized(True)
            pt.set_fixed(False)
            opt.add_vertex(pt)
        
            for f in p.frames:
                edge = g2o.EdgeProjectP2MC()
                edge.set_vertex(0, pt)
                edge.set_vertex(1, opt.vertex(f.id))
                uv = f.kpus[f.pts.index(p)]
                edge.set_measurement(uv)
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)

        # opt.set_verbose(True)
        opt.initialize_optimization()
        opt.optimize(80)

        # put frames back
        for f in self.frames:
            est = opt.vertex(f.id).estimate()
            R = est.rotation().matrix()
            t = est.translation()
            f.pose = poseRt(R, t)

        # put points back
        new_points = []
        for p in self.points:
            vert = opt.vertex(p.id + PT_ID_OFFSET)
            if vert is None:
                new_points.append(p)
                continue
            est = vert.estimate()
            old_point = len(p.frames) == 2 and p.frames[-1] not in local_frames
            errs = []
            for f in p.frames:
                uv = f.kpus[f.pts.index(p)]
                proj = np.dot(f.k, est)
                proj = proj[0:2] / proj[2]
                errs.append(np.linalg.norm(proj-uv))
            #if (old_point and np.mean(errs) > 30) or np.mean(errs) > 100:
            #    p.delete()
            #    continue
            p.pt = np.array(est)
            new_points.append(p)
        self.points = new_points
        return opt.chi2()

    # *** viewer ***

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
        pangolin.CreateWindowAndBind('3D Map', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
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

        # draw pose
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawCameras(self.state[0])

        # draw keypoints
        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(self.state[1], self.state[2])

        pangolin.FinishFrame()

    def display(self):
        poses, pts, colors = [], [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
            colors.append(p.color)
        self.q.put((np.array(poses), np.array(pts), np.array(colors)/256.0))
