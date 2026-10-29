"""Render the paused DiffusionPDE run's complete recorded loss history."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V4 = Path(__file__).resolve().parents[1]
ROOT = V4.parents[3]
RUN = ROOT / 'cases/turbulent_combustion/runs/tc_diffusion_pde_5000ep/20261027T222333Z_187a0271'

def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

def main():
    train = rows(RUN/'metrics/history.jsonl')
    val = rows(RUN/'metrics/validation_history.jsonl')
    fig, ax = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    ax[0].plot([r['epoch'] for r in train],[r['edm_weighted_mse'] for r in train],lw=1,label='Training epoch mean')
    ax[0].plot([r['epoch'] for r in val],[r['validation_loss'] for r in val],lw=1,label='Validation sampled loss')
    ax[1].plot([r['epoch'] for r in train],[r['edm_clean_mse'] for r in train],lw=1,label='Training epoch mean')
    ax[1].plot([r['epoch'] for r in val],[r['components']['edm_clean_mse'] for r in val],lw=1,label='Validation sampled loss')
    for a,title in zip(ax,['EDM weighted denoising loss','EDM clean-field MSE']):
        a.axvline(1400,color='black',ls='--',lw=1,label='last.pt: epoch 1400'); a.set(xlabel='Epoch',ylabel='Loss',title=title,yscale='log'); a.legend(fontsize=8); a.grid(alpha=.2)
    fig.suptitle('Paused DiffusionPDE history: logged training through epoch %s' % train[-1]['epoch'])
    out=V4/'figures'/'diffusion_paused_loss';fig.savefig(out.with_suffix('.png'),dpi=180);fig.savefig(out.with_suffix('.pdf'));plt.close(fig)
    record={'run':str(RUN),'checkpoint_epoch':1400,'training_last_epoch':train[-1]['epoch'],
            'training_last_step':train[-1]['step'],'validation_last_epoch':val[-1]['epoch'],
            'validation_last_step':val[-1]['step'],'training_history':str(RUN/'metrics/history.jsonl'),
            'validation_history':str(RUN/'metrics/validation_history.jsonl'),
            'description':'Training epoch means and sampled validation loss; history beyond the checkpoint is not checkpointed state.',
            'png':'figures/diffusion_paused_loss.png','pdf':'figures/diffusion_paused_loss.pdf'}
    (V4/'metadata/paused_loss_curve.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Training last epoch',train[-1]['epoch'],'validation last epoch',val[-1]['epoch'],'checkpoint epoch 1400')
if __name__=='__main__':main()
