# Verification checklist

- [ ] `python train.py --smoke-test` passes.
- [ ] `python train.py --model vae --limit 4096 --epochs 2` creates the VAE checkpoint, CSV, and three figures.
- [ ] `python train.py --model gan --limit 4096 --epochs 2` creates the DCGAN checkpoint, CSV, and sample figure.
- [ ] The report includes final losses and visual observations from the generated figures.
- [ ] Before claiming production readiness, evaluate with FID/KID and test on a licensed higher-resolution dataset.
