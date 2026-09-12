"""Single-GPU long-run comparison entrypoint with validation stopping and time cap.
Config: path or ppo; trainer.long_budget_seconds=0 disables the optional time cap. No fixed max steps.
"""
import json,os,time
from pathlib import Path
import hydra,wandb
import torch.distributed as dist
from RL2.trainer.path import PathTrainer
from RL2.trainer.ppo import PPOTrainer
from RL2.utils.checkpointing import load_ckpt,save_ckpt,save_model
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.validation_stop import record_validation
from experiments.long_backup import publish_snapshot


def atomic_json(p,value):
    tmp=p.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def train(trainer, start_time):
    cfg=trainer.config
    assert dist.get_world_size()==1, 'Long-run accounting currently requires single GPU'
    budget=float(cfg.trainer.long_budget_seconds)
    assert budget>=0 and cfg.trainer.test_freq==20
    is_path=isinstance(trainer,PathTrainer)
    if not is_path:
        assert cfg.adv.estimator=='reinforce' and cfg.actor.kl.coef==0
    workers=(trainer.actor,None,trainer.rollout)
    step=load_ckpt(trainer,workers)
    root=Path(cfg.trainer.save_dir);root.mkdir(parents=True,exist_ok=True)
    statefile=root/'long_state.json'
    state=json.loads(statefile.read_text()) if statefile.exists() else {'history':[],'used_seconds':0,'generated_tokens':0,'rollout_responses':0}
    state['history']=[r for r in state['history'] if r['step']<=step]
    previously_used=state['used_seconds']
    def used():return previously_used+time.monotonic()-start_time
    def persist():
        state.update(step=step,used_seconds=used());atomic_json(statefile,state)
    def evaluate():
        scores=[]
        for data in trainer.test_dataloader:
            trainer.rollout(data,False,step)
            scores.append(trainer.rollout.last_metrics['scores/test'])
        assert len(scores)==1, 'Expected one fixed 500-question validation batch'
        state['history'],stop,improved=record_validation(state['history'],step,scores[0])
        if scores[0] > state.get('best_accuracy', -1.0) or not (root/'best/latest/config.json').exists():
            previous_dir=cfg.trainer.save_dir
            trainer.completed_step=step
            cfg.trainer.save_dir=str(root/'best')
            try:save_model(trainer,trainer.actor)
            finally:cfg.trainer.save_dir=previous_dir
            state['best_step']=step;state['best_accuracy']=scores[0]
        persist()
        if step > 100 or not is_path:
            publish_snapshot(root)
        return stop
    reason=None
    if not state['history'] or state['history'][-1]['step']!=step:
        if evaluate():reason='validation_below_previous_two'
    try:
        while reason is None:
            if budget and used()>=budget:reason='time_budget';break
            for data in trainer.train_dataloader:
                if trainer.should_stop():reason='user_pause';break
                if budget and used()>=budget:reason='time_budget';break
                step+=1
                td,cu=trainer.rollout(data,True,step)
                state['generated_tokens']+=int(td['action_mask'].sum().item())
                state['rollout_responses']+=int(td['action_mask'].shape[0])
                if not is_path:td=trainer.actor.compute_logps(td,step)
                trainer.compute_advantages(td,cu,step)
                if is_path:trainer.actor.update_path(td,step)
                else:trainer.actor.update_original(td,step)
                save_ckpt(trainer,(trainer.actor,None),step)
                if trainer.should_stop():reason='user_pause';break
                trainer.rollout.update(trainer.actor,step)
                if step%20==0 and evaluate():reason='validation_below_previous_two';break
                persist()
            if reason is not None:break
        save_ckpt(trainer,(trainer.actor,None),step,force=True)
        state['termination_reason']=reason;persist()
        # Best weights already exist; never label last-step weights as best.
        if reason!='user_pause':
            atomic_json(root/'long_completed.json',state)
        publish_snapshot(root)
        if cfg.trainer.use_wandb:
            wandb.summary.update({'termination_reason':reason,'best_validation':state.get('best_accuracy'),'best_step':state.get('best_step'),'total_generated_tokens':state['generated_tokens']})
    finally:persist()


@hydra.main(config_path='../RL2/trainer/config',config_name='path',version_base=None)
def main(cfg):
    start=time.monotonic();initialize_global_process_group()
    cls=PathTrainer if hasattr(cfg.actor,'path') else PPOTrainer
    trainer=cls(cfg)
    try:
        train(trainer,start)
        if cfg.trainer.use_wandb:wandb.finish()
    finally:
        if cfg.trainer.use_wandb:wandb.teardown()
        trainer.rollout.llm.shutdown();dist.destroy_process_group()

if __name__=='__main__':main()
