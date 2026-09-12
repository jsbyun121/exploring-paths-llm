import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from RL2.utils.checkpointing import save_ckpt, load_ckpt, completed_checkpoints, optimizer_load_template


class Loader:
    def __init__(self):
        self.cursor = 0
    def state_dict(self):
        return {"cursor": self.cursor}
    def load_state_dict(self,state):
        self.cursor = state["cursor"]


class RecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.group_dir = tempfile.TemporaryDirectory()
        dist.init_process_group("gloo", init_method="file://" + cls.group_dir.name + "/group", rank=0, world_size=1)
    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.group_dir.cleanup()
    def trainer(self, root):
        return SimpleNamespace(config=OmegaConf.create({"trainer": {
            "save_dir":root,"save_freq":10,"keep_checkpoints":2,"load_ckpt_from":"latest"}}),
            train_dataloader=Loader())
    def test_roundtrip_rng_loader_and_retention(self):
        with tempfile.TemporaryDirectory() as root:
            t = self.trainer(root)
            for step in [10,20,30]:
                t.train_dataloader.cursor = step
                save_ckpt(t, (), step)
            self.assertEqual([p.name for p in completed_checkpoints(root)], ["step20","step30"])
            expected = (random.random(), torch.rand(3))
            t.train_dataloader.cursor = 99
            self.assertEqual(load_ckpt(t,()),30)
            self.assertEqual(t.train_dataloader.cursor,30)
            self.assertEqual(random.random(),expected[0])
            torch.testing.assert_close(torch.rand(3),expected[1],atol=0,rtol=0)
    def test_failed_save_does_not_publish_or_delete_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            t=self.trainer(root)
            save_ckpt(t,(),10)
            with patch("RL2.utils.checkpointing.dcp.save",side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_ckpt(t,(),20)
            self.assertEqual([p.name for p in completed_checkpoints(root)],["step10"])
    def test_legacy_checkpoint_and_missing_resume(self):
        with tempfile.TemporaryDirectory() as root:
            t=self.trainer(root)
            with self.assertRaises(FileNotFoundError):
                load_ckpt(t,())
            dcp.save({"step":100,"dataloader":{"cursor":100}},checkpoint_id=str(Path(root)/"step100"))
            with self.assertWarnsRegex(UserWarning,"Legacy checkpoint"):
                self.assertEqual(load_ckpt(t,()),100)

    def test_fresh_adam_restores_moments_and_matches_next_update(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as root:
            model = torch.nn.Linear(3,2)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
            x = torch.randn(4,3)
            for _ in range(3):
                optimizer.zero_grad();model(x).square().sum().backward();optimizer.step()
            dcp.save({"worker0":{"model":model.state_dict(),"optimizer":optimizer.state_dict()}},checkpoint_id=root)
            recovered = torch.nn.Linear(3,2)
            fresh = torch.optim.AdamW(recovered.parameters(),lr=.01)
            metadata = dcp.FileSystemReader(root).read_metadata()
            template={"worker0":{"model":recovered.state_dict(),
                       "optimizer":optimizer_load_template(fresh,metadata,"worker0")}}
            dcp.load(template,checkpoint_id=root)
            recovered.load_state_dict(template["worker0"]["model"])
            fresh.load_state_dict(template["worker0"]["optimizer"])
            self.assertEqual(len(fresh.state),2)
            for state in fresh.state.values():
                self.assertEqual(state["step"].item(),3)
            for m,o in [(model,optimizer),(recovered,fresh)]:
                o.zero_grad();m(x).square().sum().backward();o.step()
            for a,b in zip(model.parameters(),recovered.parameters()):
                torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__ == "__main__":
    unittest.main()
