import unittest
from unittest.mock import patch
from math import ceil
from HPCSub.config import *
from HPCSub.pbs import *
from HPCSub.pbs import placement_policy


class PbsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.queue_info = Queues()

    @patch('HPCSub.pbs.is_admin', False)
    def test_valid_queue(self):
        dyna_queue = {str(x) for x in
                      self.queue_info.valid_queues('dyna', use_gpu=False, user_group=frozenset(['safety_hf']))}
        self.assertEqual(dyna_queue, {'amd', 'amd2', 'debug'})
        starccm_gpu = {str(x) for x in self.queue_info.valid_queues('starccm', use_gpu=True)}
        self.assertEqual(starccm_gpu, {'gpu'})
        dyna_compress = {str(x) for x in self.queue_info.valid_queues('dyna_compress')}
        self.assertEqual(dyna_compress, {'post'})

    def test_recommend_cores(self):
        for cores in (6, 12, 18, 30, 60, 90):
            cpu, _, queue = self.queue_info.recommend(all_app_config['starccm'], cpu=cores, queue='fy_cfd')
            self.assertFalse(queue.has_gpu)
            if cores <= 8:
                self.assertEqual(cpu, 8)
            if cores < 32:
                self.assertTrue(cpu % 8 == 0)
            elif cores < 64:
                self.assertTrue(cpu == 64)
            else:
                self.assertTrue(cpu % 64 == 0)

        cpu, _, queue = self.queue_info.recommend(all_app_config['custom'], cpu=3)
        self.assertEqual(cpu, 3)

        for app in all_app_config.values():
            try:
                cpu, _, queue = self.queue_info.recommend(app)
                self.assertGreaterEqual(cpu, app.DefaultMinCores)
                self.assertIn(app.Name, queue.Apps)
            except SubError:
                for q in self.queue_info.values():
                    self.assertNotIn(app.Name, q.Apps)

    def test_gpu(self):
        gpu, gpu_opt, q = self.queue_info.recommend(all_app_config['nano'])
        self.assertEqual(gpu, all_app_config['nano'].DefaultGPU)
        self.assertEqual(gpu_opt, all_app_config['nano'].DefaultCoreWithGPU)
        self.assertEqual(q.Name, 'gpu')
        gpu, gpu_opt, q = self.queue_info.recommend(all_app_config['starccm'], gpu=6)
        self.assertEqual(gpu, 6)
        self.assertEqual(gpu_opt, 1)
        self.assertEqual(q.Name, 'gpu')

    def test_mpi_policy(self):
        for cores in (8, 16, 32, 64, 128):
            for omp in (0, -1, -2, -3, 65):
                try:
                    s, p, v = placement_policy(self.queue_info['amd'], cores=cores, gpu_opt=0, mpi=True, openmp_opt=omp)
                except SubError as e:
                    self.assertTrue(omp in (-2, -3, 65))
                    continue
                if omp == 0:
                    self.assertEqual(s.ncpus, s.mpiprocs)
                elif omp == -1:
                    self.assertEqual(s.mpiprocs, 1)
                    self.assertEqual(s.ompthreads, min(cores, 64))
                elif omp == -2:
                    self.assertEqual(s.ompthreads, 32)
                elif omp == -3:
                    self.assertEqual(s.ompthreads, 8)
                self.assertEqual(v['SUB_CPU'], s.select * s.ncpus)
                if cores <= 8:
                    self.assertEqual(p.group, ResourceGroup.Numa)
                    self.assertEqual(s.select, 1)
                elif cores <= 32:
                    self.assertEqual(p.group, ResourceGroup.Socket)
                    self.assertEqual(s.select, 1)
                elif cores <= 64:
                    self.assertEqual(p.arrangement, Arrangement.Pack)
                    self.assertEqual(s.select, 1)
                else:
                    self.assertEqual(p.arrangement, Arrangement.Free)
                    self.assertEqual(p.group, ResourceGroup.IBSwitch)
                    self.assertEqual(s.select, ceil(cores / 64))
