import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scripts.benchmark_models import scfm_loss
from scripts.metrics import w1_euclidean,vendi_cosine,paired_ssim,soft_nmi_variants
from scripts.prepare_official_data import load_arrays
from scripts.run_official import load_author_modules,build_continuous,make_model,defaults,AUTHOR_SHA


class ProtocolTests(unittest.TestCase):
    def test_data_counts_and_fixed_splits(self):
        with tempfile.TemporaryDirectory() as d:
            a=load_arrays('pinwheel',d,2026,'full');b=load_arrays('pinwheel',d,2026,'full')
        self.assertEqual([len(a[k][0]) for k in ['train','val','test']],[15000,2000,2000])
        self.assertEqual(a['metadata'],b['metadata'])
        self.assertEqual(len(set(a['metadata']['data_sha256'].values())),3)

    def test_metrics_known_cases(self):
        self.assertAlmostEqual(w1_euclidean(np.array([[0,0],[2,0]]),np.array([[1,1],[3,1]])),2**.5)
        self.assertAlmostEqual(vendi_cosine(np.eye(4)),4)
        self.assertAlmostEqual(vendi_cosine(np.ones((4,2))),1)
        x=np.random.default_rng(2).random((2,1,8,8))
        self.assertAlmostEqual(paired_ssim(x,x),1)
        nmi=soft_nmi_variants(np.eye(3),np.arange(3))
        self.assertAlmostEqual(nmi['soft_nmi_arithmetic_corrected'],1)
        self.assertAlmostEqual(nmi['soft_nmi_paper_printed'],.5)

    def test_estimator_draw_order_and_gradients(self):
        class Posterior:
            def __init__(self,draws):self.draws=iter(draws)
            def rsample(self):return next(self.draws)
        class Q:
            def __init__(self,draws):self.q=Posterior(draws)
            def __call__(self,x,t):return self.q
        class Field(nn.Module):
            def __init__(self):super().__init__();self.a=nn.Parameter(torch.tensor(2.))
            def forward(self,t,x,z):return self.a*z
        x=torch.zeros(1,1);fixed=(torch.zeros(1),x,x)
        v=Field();loss=scfm_loss(v,Q([torch.tensor([[1.]]),torch.tensor([[-1.]]),torch.tensor([[1.]]),torch.tensor([[-1.]])]),x,'k4',fixed_bridge=fixed)
        self.assertEqual(loss.item(),0) # mean-square of draws would incorrectly give 4
        v=Field();loss=scfm_loss(v,Q([torch.tensor([[1.]]),torch.tensor([[-1.]])]),x,'cross',fixed_bridge=fixed)
        self.assertEqual(loss.item(),-4);loss.backward();self.assertEqual(v.a.grad.item(),-4)

    def test_author_k1_matches_extension(self):
        m=load_author_modules();self.assertEqual(m['revision'],AUTHOR_SHA)
        vt,rt=build_continuous((2,),1,width=16,depth=1)
        prior=torch.distributions.Normal(torch.zeros(1),torch.ones(1))
        official=m['sfa'].FlowMatchingLoss(vt,rt,prior,alpha=.01,sig_min=1e-4,fixz=True)
        x=torch.randn(8,2)
        torch.manual_seed(7);a=official(x)
        torch.manual_seed(7);b=scfm_loss(vt,rt,x,'k1',alpha=.01,sigma=1e-4)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

    def test_all_author_model_backward(self):
        for dataset,shape in [('pinwheel',(2,)),('pinwheel_mixture',(2,)),('mnist',(1,28,28)),
                              ('mnist_mixture',(1,28,28)),('hvg',(5000,)),('pendulum',(3,1,10,10))]:
            with self.subTest(dataset=dataset):
                cfg=defaults(dataset,'pilot');cfg.update(width=16,depth=1,sequence_length=3)
                model,loss=make_model(dataset,shape,cfg)
                x=torch.randn(3,2,1,10,10) if dataset=='pendulum' else torch.randn(4,*shape)
                value=loss(x);self.assertTrue(torch.isfinite(value));value.backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads);self.assertTrue(all(torch.isfinite(g).all() for g in grads))


if __name__=='__main__':unittest.main()
