import copy, json, math, os, platform, random, struct, time
from pathlib import Path
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F

SEED=7
random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(min(4,os.cpu_count() or 1))
ART=Path('artifacts'); ART.mkdir(exist_ok=True)

PROSE=("""Training loops should tell the truth. A model sees tokens, produces logits, compares predictions with targets, and follows gradients. Small implementation details can silently change what objective is optimized. Good instrumentation turns hidden assumptions into numbers we can inspect.
Gradient accumulation is mathematically simple only when every micro-batch contributes the same number of valid tokens. When sequence lengths differ, averaging already-averaged losses gives each micro-batch equal weight instead of each token equal weight. That changes the gradient.
The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. We train small language models to make every tensor and gradient observable.
""")*25
DIAG=("""ERROR:: retry=3; code=E17; packet=[x,y,z]; status=FAIL; checksum=9f2a. >>> ASSERT_FALSE <<<
if (gradient != expected) { inspect(shape); inspect(mask); inspect(scale); }
%%%% 000111000111 ::: short noisy diagnostic stream ::: ####
""")*40
chars=sorted(set(PROSE+DIAG)); stoi={c:i for i,c in enumerate(chars)}; V=len(chars)
def enc(s): return torch.tensor([stoi[c] for c in s],dtype=torch.long)
prose,diag,full=enc(PROSE),enc(DIAG),enc(PROSE+DIAG)
def batch(data,B,T):
    starts=torch.randint(0,len(data)-T-1,(B,)); x=torch.stack([data[i:i+T] for i in starts]); y=torch.stack([data[i+1:i+T+1] for i in starts]); return x,y

class TinyTruthGPT(nn.Module):
    def __init__(self,vocab=V,C=32,H=4,Ff=64,Tmax=64):
        super().__init__(); self.vocab_size=vocab; self.d_model=C; self.n_heads=H; self.head_dim=C//H; self.d_ff=Ff
        self.tok=nn.Embedding(vocab,C); self.pos=nn.Embedding(Tmax,C); self.ln1=nn.LayerNorm(C); self.qkv=nn.Linear(C,3*C,bias=False); self.proj=nn.Linear(C,C,bias=False); self.ln2=nn.LayerNorm(C); self.fc1=nn.Linear(C,Ff); self.fc2=nn.Linear(Ff,C); self.head=nn.Linear(C,vocab,bias=False)
    def forward(self,tokens,record=False):
        B,T=tokens.shape; C,H,D=self.d_model,self.n_heads,self.head_dim; s={}
        def r(n,x): s[n]=tuple(x.shape); return x
        r('tokens',tokens); te=r('token_embedding',self.tok(tokens)); ids=r('position_ids',torch.arange(T,device=tokens.device)); pe=r('position_embedding',self.pos(ids)); x=r('embedded_sum',te+pe)
        z=r('ln1',self.ln1(x)); p=r('qkv_packed',self.qkv(z)); q,k,v=p.chunk(3,-1); r('q',q); r('k',k); r('v',v)
        qh=r('q_heads',q.view(B,T,H,D).transpose(1,2)); kh=r('k_heads',k.view(B,T,H,D).transpose(1,2)); vh=r('v_heads',v.view(B,T,H,D).transpose(1,2))
        scores=r('attention_scores',(qh@kh.transpose(-2,-1))/math.sqrt(D)); mask=r('causal_mask',torch.triu(torch.ones(T,T,dtype=torch.bool,device=tokens.device),1)); a=r('attention_weights',F.softmax(scores.masked_fill(mask,float('-inf')),-1)); ch=r('context_heads',a@vh); ctx=r('context',ch.transpose(1,2).contiguous().view(B,T,C)); ao=r('attention_output',self.proj(ctx)); x=r('residual_after_attention',x+ao)
        z=r('ln2',self.ln2(x)); h=r('mlp_hidden',self.fc1(z)); ga=r('mlp_activation',F.gelu(h)); mo=r('mlp_output',self.fc2(ga)); x=r('residual_after_mlp',x+mo); logits=r('logits',self.head(x)); return (logits,s) if record else logits

def lm_loss(m,x,y,reduction='mean'):
    z=m(x); B,T,Vv=z.shape; return F.cross_entropy(z.reshape(B*T,Vv),y.reshape(B*T),reduction=reduction)

# 1) Shape truth + one real training step
model=TinyTruthGPT(); PARAM_COUNT=sum(p.numel() for p in model.parameters()); x,y=batch(prose,4,24); logits,shapes=model(x,True)
axis={'tokens':'[B,T] input token ids','token_embedding':'[B,T,C] token vectors','position_ids':'[T] positions','position_embedding':'[T,C] position vectors','embedded_sum':'[B,T,C] summed embeddings','ln1':'[B,T,C] normalized residual','qkv_packed':'[B,T,3C] packed projections','q':'[B,T,C] queries','k':'[B,T,C] keys','v':'[B,T,C] values','q_heads':'[B,H,T,D] query heads','k_heads':'[B,H,T,D] key heads','v_heads':'[B,H,T,D] value heads','attention_scores':'[B,H,T,T] query-key scores','causal_mask':'[T,T] future mask','attention_weights':'[B,H,T,T] probabilities','context_heads':'[B,H,T,D] attended values','context':'[B,T,C] concatenated heads','attention_output':'[B,T,C] projected attention','residual_after_attention':'[B,T,C] residual','ln2':'[B,T,C] normalized residual','mlp_hidden':'[B,T,F] MLP expansion','mlp_activation':'[B,T,F] GELU','mlp_output':'[B,T,C] MLP projection','residual_after_mlp':'[B,T,C] final hidden','logits':'[B,T,V] vocabulary scores'}
print('B=batch T=sequence C=model width H=heads D=head width F=MLP width V=vocab')
for n,shape in shapes.items(): print(f'{n:26s} {str(shape):18s} {axis[n]}')
opt=torch.optim.SGD(model.parameters(),lr=.01); opt.zero_grad(); flat_logits=logits.reshape(-1,V); flat_targets=y.reshape(-1); loss=F.cross_entropy(flat_logits,flat_targets); loss.backward(); print('targets',tuple(y.shape),'flat_logits',tuple(flat_logits.shape),'flat_targets',tuple(flat_targets.shape),'loss',tuple(loss.shape))
for n,p in model.named_parameters(): print(f'{n:28s} param={tuple(p.shape)} grad={tuple(p.grad.shape)}')
opt.step(); opt.zero_grad(set_to_none=True)

# 2) Finite-difference gradient check
check=copy.deepcopy(model).double(); cx,cy=batch(prose,2,8); check.zero_grad(); L=lm_loss(check,cx,cy); L.backward(); row,col=49,26; autograd=check.head.weight.grad[row,col].item(); eps=1e-5
with torch.no_grad():
    w0=check.head.weight[row,col].item(); check.head.weight[row,col]=w0+eps; lp=lm_loss(check,cx,cy).item(); check.head.weight[row,col]=w0-eps; lm=lm_loss(check,cx,cy).item(); check.head.weight[row,col]=w0
finite=(lp-lm)/(2*eps); abs_err=abs(autograd-finite); rel_err=abs_err/(abs(autograd)+1e-12); print('gradient check',autograd,finite,rel_err); assert rel_err<1e-6

# 3) Break accumulation and plot wrong vs correct curves
torch.manual_seed(SEED+1); base=TinyTruthGPT(); wrong,correct=copy.deepcopy(base),copy.deepcopy(base); ow=torch.optim.SGD(wrong.parameters(),lr=.18); oc=torch.optim.SGD(correct.parameters(),lr=.18); esx,esy=batch(diag,8,12); elx,ely=batch(prose,8,48)
def eval_loss(m):
    with torch.no_grad(): return ((lm_loss(m,esx,esy,'sum')+lm_loss(m,elx,ely,'sum'))/(esy.numel()+ely.numel())).item()
wc,cc=[],[]
for _ in range(80):
    sx,sy=batch(diag,8,12); lx,ly=batch(prose,8,48)
    ow.zero_grad(); ((lm_loss(wrong,sx,sy)+lm_loss(wrong,lx,ly))/2).backward(); ow.step()
    oc.zero_grad(); ((lm_loss(correct,sx,sy,'sum')+lm_loss(correct,lx,ly,'sum'))/(sy.numel()+ly.numel())).backward(); oc.step(); wc.append(eval_loss(wrong)); cc.append(eval_loss(correct))
wrong_final,correct_final=wc[-1],cc[-1]; gap=wrong_final-correct_final; print('accumulation',esy.numel(),ely.numel(),wrong_final,correct_final,gap)
plt.figure(figsize=(8,4)); plt.plot(wc,label='WRONG: avg micro-batch means'); plt.plot(cc,label='CORRECT: token weighted'); plt.xlabel('optimizer update'); plt.ylabel('fixed token-weighted loss'); plt.legend(); plt.tight_layout(); plt.savefig(ART/'gradient_accumulation_wrong_vs_correct.png',dpi=150); plt.close()

# Exact reference-gradient proof on same-distribution unequal lengths
torch.manual_seed(SEED+11); pb=TinyTruthGPT(); psx,psy=batch(prose,8,12); plx,ply=batch(prose,8,48); nt=psy.numel()+ply.numel()
def flatgrad(m): return torch.cat([p.grad.reshape(-1).float() for p in m.parameters() if p.grad is not None])
r=copy.deepcopy(pb); r.zero_grad(); ((lm_loss(r,psx,psy,'sum')+lm_loss(r,plx,ply,'sum'))/nt).backward(); gr=flatgrad(r)
c=copy.deepcopy(pb); c.zero_grad(); (lm_loss(c,psx,psy,'sum')/nt).backward(); (lm_loss(c,plx,ply,'sum')/nt).backward(); gc=flatgrad(c)
w=copy.deepcopy(pb); w.zero_grad(); (.5*lm_loss(w,psx,psy)).backward(); (.5*lm_loss(w,plx,ply)).backward(); gw=flatgrad(w)
def cmp(g): return F.cosine_similarity(g,gr,dim=0).item(),(g-gr).abs().max().item(),((g-gr).norm()/(gr.norm()+1e-12)).item()
ccos,cmax,crel=cmp(gc); wcos,wmax,wrel=cmp(gw); print('reference proof',ccos,crel,wcos,wrel); assert ccos>.999999 and crel<1e-6 and wrel>1e-3
plt.figure(figsize=(6,4)); plt.bar(['correct','broken'],[max(crel,1e-9),wrel]); plt.yscale('log'); plt.ylabel('relative L2 gradient error'); plt.tight_layout(); plt.savefig(ART/'gradient_reference_proof.png',dpi=150); plt.close()

# 4) Grad norm trace
def gnorm(m): return math.sqrt(sum(p.grad.detach().float().norm().item()**2 for p in m.parameters() if p.grad is not None))
torch.manual_seed(SEED+2); tm=TinyTruthGPT(); to=torch.optim.AdamW(tm.parameters(),lr=3e-3); px,py=batch(prose,8,32); probe_losses=[]; grad_norms=[]
for step in range(90):
    bx,by=batch(full,8,32); to.zero_grad(); tl=lm_loss(tm,bx,by); tl.backward(); grad_norms.append(gnorm(tm)); to.step()
    with torch.no_grad(): probe_losses.append(lm_loss(tm,px,py).item())
cands=[]
for i in range(1,86):
    gm=abs(grad_norms[i]-grad_norms[i-1])/(abs(grad_norms[i-1])+1e-12); lm0=abs(probe_losses[i]-probe_losses[i-1])/(abs(probe_losses[i-1])+1e-12); future=max(abs(probe_losses[j]-probe_losses[i])/(abs(probe_losses[i])+1e-12) for j in range(i+1,i+4)); cands.append(((gm/(lm0+1e-4))*future,i,gm,lm0,future))
_,lead_step,lead_grad,lead_loss,lead_future=max(cands); print('leading signal',lead_step,lead_grad,lead_loss,lead_future)
fig,ax=plt.subplots(figsize=(8,4)); ax.plot(probe_losses,label='probe loss'); ax.axvline(lead_step,ls='--'); ax2=ax.twinx(); ax2.plot(grad_norms,alpha=.6,label='grad norm'); fig.tight_layout(); fig.savefig(ART/'grad_norm_leading_signal.png',dpi=150); plt.close(fig)

# 5) MFU: explicit numerator and denominator
def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
def bench(m,B=16,T=32,warm=4,iters=12):
    o=torch.optim.SGD(m.parameters(),lr=1e-3); bx,by=batch(prose,B,T); dev0=next(m.parameters()).device; bx,by=bx.to(dev0),by.to(dev0)
    for _ in range(warm): o.zero_grad(); l=lm_loss(m,bx,by); l.backward(); o.step()
    sync(); ts=[]
    for _ in range(3):
        t=time.perf_counter()
        for _ in range(iters): o.zero_grad(); l=lm_loss(m,bx,by); l.backward(); o.step()
        sync(); ts.append((time.perf_counter()-t)/iters)
    return sorted(ts)[1],bx.numel(),T
def roofline():
    text=Path('/proc/cpuinfo').read_text(errors='ignore') if Path('/proc/cpuinfo').exists() else ''; flags=set(); mhz=2300.; cpu=platform.processor() or 'CPU'
    for line in text.splitlines():
        if line.startswith('model name'): cpu=line.split(':',1)[1].strip()
        elif line.startswith('flags') and not flags: flags=set(line.split(':',1)[1].split())
        elif line.startswith('cpu MHz'): mhz=float(line.split(':',1)[1]); break
    lanes=16 if 'avx512f' in flags else 8 if ('avx2' in flags or 'avx' in flags) else 4; units=2 if 'fma' in flags else 1; peak0=torch.get_num_threads()*mhz*1e6*lanes*2*units/1e12; return peak0,f'analytical allocated-CPU FP32 roofline: {torch.get_num_threads()} threads × {mhz/1000:.3f} GHz × {lanes} fp32 SIMD lanes × 2 FLOP/FMA × {units} vector issue units',{'cpu_model':cpu,'detected_mhz':mhz,'simd_fp32_lanes':lanes,'assumed_vector_issue_units_per_core':units,'pytorch_threads':torch.get_num_threads()}
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); mm=TinyTruthGPT().to(dev); C,Ff=mm.d_model,mm.d_ff
def flop_tok(T): return 3*(8*C*C+4*C*Ff+2*C*V+4*T*C)
sec,tok,Tb=bench(mm); model_tflops=flop_tok(Tb)*tok/sec/1e12; tokps=tok/sec
n=1024 if dev.type=='cpu' else 4096; a=torch.randn(n,n,device=dev); b=torch.randn(n,n,device=dev); [a@b for _ in range(2)]; sync(); t=time.perf_counter(); reps=5; [a@b for _ in range(reps)]; sync(); empirical=(2*n**3)/((time.perf_counter()-t)/reps)/1e12
THEORETICAL_PEAK_TFLOPS=None
if THEORETICAL_PEAK_TFLOPS: peak=float(THEORETICAL_PEAK_TFLOPS); peak_src='configured vendor peak'; kind='canonical accelerator MFU'; details=None
elif dev.type=='cpu': peak,peak_src,details=roofline(); kind='analytical allocated-CPU roofline MFU estimate'
else: peak=None; peak_src='set THEORETICAL_PEAK_TFLOPS'; kind='canonical MFU unavailable without vendor peak'; details=None
reported_mfu=model_tflops/peak if peak else None; gap40=(.40-reported_mfu)*100 if reported_mfu is not None else None; empirical_util=model_tflops/empirical
sweep=[]
for B in [4,8,16,32,64]:
    sm=TinyTruthGPT().to(dev); s,tk,Ts=bench(sm,B=B); tf=flop_tok(Ts)*tk/s/1e12; sweep.append({'batch_size':B,'seconds_per_step':s,'tokens_per_second':tk/s,'tflops':tf,'mfu':tf/peak if peak else None})
best=max(sweep,key=lambda z:z['tflops']); print('MFU',kind,reported_mfu,'gap40',gap40,'best',best)
plt.figure(figsize=(6,4)); plt.plot([z['batch_size'] for z in sweep],[z['tflops'] for z in sweep],marker='o'); plt.xscale('log',base=2); plt.xlabel('batch size'); plt.ylabel('model TFLOP/s'); plt.tight_layout(); plt.savefig(ART/'mfu_batch_sweep.png',dpi=150); plt.close()

# 6) Decimal 0.1 in fp32 / bf16 / fp8 E4M3
fp32_bits=struct.unpack('>I',struct.pack('>f',.1))[0]; fp32_value=torch.tensor(.1,dtype=torch.float32).item(); bf=torch.tensor(.1,dtype=torch.bfloat16); bf16_bits=int(bf.view(torch.uint16)); bf16_value=bf.item(); fp8=torch.tensor(.1,dtype=torch.float8_e4m3fn) if hasattr(torch,'float8_e4m3fn') else None; fp8_bits=int(fp8.view(torch.uint8)) if fp8 is not None else 0x1D; fp8_value=fp8.item() if fp8 is not None else .1015625
fp32_error=abs(fp32_value-.1); bf16_error=abs(bf16_value-.1); fp8_error=abs(fp8_value-.1); print('0.1',f'{fp32_bits:032b}',f'{bf16_bits:016b}',f'{fp8_bits:08b}',fp32_error,bf16_error,fp8_error); assert fp32_bits==0x3DCCCCCD and bf16_bits==0x3DCD and fp8_bits==0x1D

summary={'schema_version':2,'seed':SEED,'vocab_size':V,'parameter_count':PARAM_COUNT,'gradient_check':{'weight':[row,col],'autograd':autograd,'finite_difference':finite,'absolute_error':abs_err,'relative_error':rel_err},'gradient_accumulation':{'short_tokens':esy.numel(),'long_tokens':ely.numel(),'wrong_final_validation_loss':wrong_final,'correct_final_validation_loss':correct_final,'wrong_minus_correct_gap':gap,'reference_gradient_proof':{'proof_short_tokens':psy.numel(),'proof_long_tokens':ply.numel(),'same_distribution':True,'correct_cosine_similarity':ccos,'correct_max_abs_error':cmax,'correct_relative_l2_error':crel,'wrong_cosine_similarity':wcos,'wrong_max_abs_error':wmax,'wrong_relative_l2_error':wrel}},'grad_norm_leading_signal':{'step':lead_step,'relative_grad_norm_move':lead_grad,'same_step_probe_loss_move':lead_loss,'next_3_step_probe_loss_move':lead_future,'trace_steps':90},'mfu':{'device':str(dev),'seconds_per_step':sec,'tokens_per_second':tokps,'estimated_model_tflops':model_tflops,'empirical_gemm_ceiling_tflops':empirical,'empirical_ceiling_utilization':empirical_util,'reported_mfu':reported_mfu,'mfu_kind':kind,'reported_peak_tflops':peak,'reported_peak_source':peak_src,'cpu_roofline_details':details,'gap_to_40_percentage_points':gap40,'batch_size_sweep':sweep,'best_sweep_batch_size':best['batch_size'],'best_sweep_tflops':best['tflops'],'best_sweep_mfu':best['mfu']},'float_0_1':{'fp32_bits':f'{fp32_bits:032b}','bf16_bits':f'{bf16_bits:016b}','fp8_e4m3fn_bits':f'{fp8_bits:08b}','fp32_value':fp32_value,'bf16_value':bf16_value,'fp8_e4m3fn_value':fp8_value,'fp32_absolute_error':fp32_error,'bf16_absolute_error':bf16_error,'fp8_e4m3fn_absolute_error':fp8_error,'recommended_training_format':'bf16'}}
(ART/'metrics.json').write_text(json.dumps(summary,indent=2))
fig,ax=plt.subplots(figsize=(9,5)); ax.axis('off'); ax.set_title('Training Loop — Experiment Summary',fontweight='bold'); lines=[f'gradient check rel. error: {rel_err:.3e}',f'correct accumulation rel-L2: {crel:.3e}',f'broken accumulation rel-L2: {wrel:.2%}',f'wrong-correct loss gap: {gap:.6f}',f'grad-norm leading step: {lead_step}',f'MFU: {reported_mfu*100:.2f}%' if reported_mfu else 'MFU peak not configured',f'0.1 bits: fp32 0x{fp32_bits:08X} / bf16 0x{bf16_bits:04X} / E4M3 0x{fp8_bits:02X}']; [ax.text(.05,.88-i*.11,t,fontsize=12,transform=ax.transAxes) for i,t in enumerate(lines)]; fig.savefig(ART/'experiment_summary.png',dpi=150,bbox_inches='tight'); plt.close(fig)
assert gap>0 and lead_grad>lead_loss and fp32_error<bf16_error<fp8_error
print(json.dumps({'gradient_relative_error':rel_err,'correct_accumulation_relative_l2':crel,'broken_accumulation_relative_l2':wrel,'loss_gap':gap,'leading_signal_step':lead_step,'reported_mfu':reported_mfu},indent=2))
