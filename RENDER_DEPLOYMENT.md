# Deploy WoundWatch on Render

`render.yaml` defines a Docker web service in Singapore, a 1 CPU / 2 GB RAM
instance, and a 1 GB persistent disk for SQLite and uploaded images. The
service loads only `models/classifier.pt` and `models/segmenter.pt`; the Docker
build excludes the other model files and all datasets. Inference runs on CPU.

## First deployment

1. Push the Render deployment files to the `main` branch of the GitHub repo.
2. In Render, choose **New > Blueprint**, connect
   `hitarthkothari9641-coder/WoundWatch`, select `main`, and apply the Blueprint
   from `render.yaml`. Render account access is needed for this one-time step.
3. Wait for the deploy to become live, then open the generated
   `https://woundwatch.onrender.com` URL. Render sets
   `RENDER_EXTERNAL_HOSTNAME` automatically; the service uses it for host
   validation and trusts Render's forwarded HTTPS scheme.
4. Check `/live` for process health and `/health` for loaded-model health.
   `/health` returns success only after both checkpoints load.
5. Create clinician accounts through the Render service Shell. In the shell:

   ```sh
   cd /app
   export PYTHONPATH=/app/python
   export WOUNDWATCH_DATA_DIR=/var/data
   python -m service.manage create-user --email clinician@example.org --name "Clinician" --role clinician
   ```

   Enter a unique password when prompted. Do not use demo credentials or a
   shared default password. Patients can register through the application.

## Custom domain and operations

Add a custom domain in the Render service settings. Add that hostname to the
`WOUNDWATCH_ALLOWED_HOSTS` environment variable as a comma-separated value if
you want it accepted alongside Render's generated hostname. Redeploy after
changing environment variables.

The persistent disk keeps SQLite and image uploads across deploys. It is
single-instance storage, and each deploy can briefly interrupt service. The
disk starts at 1 GB and can be increased later; Render does not support
shrinking it. Set an image retention policy and monitor disk use before
accepting ongoing uploads. Use Render's disk snapshots plus a separate tested
backup process for recovery.

## Release limits

This is a research prototype, not a clinically validated diagnostic service.
Do not use its result as a diagnosis or treatment decision. The available
training and validation data are small, lack patient-level grouping, and do
not cover skin-tone and camera subgroups adequately. Do not put identifiable
patient photos or health information on a public hosting account until your
organization has reviewed consent, privacy, legal, retention, access-control,
and vendor requirements. A persistent disk and HTTPS do not by themselves
make this service suitable for regulated clinical data.

Confirm that dataset and checkpoint licenses allow hosted use before
deployment. The service is internet reachable and should be treated as a
controlled research pilot; provision named clinician accounts before inviting
any users.
