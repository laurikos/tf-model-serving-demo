# Linux deployment

These files were deployed and verified on the Ubuntu 26.04 Linux/amd64 replacement
on 2026-11-01. The old server remains running and DNS remains unchanged. For another
deployment, first preserve and verify an off-server backup of the old application,
model, proxy, and TLS state.

Verified behavior: HTTPS routes for both demos; PNG/JPEG inference and misleading
filenames; 400/413 upload failures; an eight-request burst with one 200 and seven
429 responses; 60 additional successful inferences; controlled restarts and readiness
recovery. External connections to ports 9000, 9999, 8500, and 8501 were refused.

The model cgroup peaked at 387,866,624 bytes (370 MiB), settled near 375 MB, and had
zero OOM events during the test. The HTTP service peaked at 67,104,768 bytes (64 MiB).
Native model tasks peaked at 266 during startup and settled at 15. Mean repeated
inference latency was 0.219 seconds on the 1-vCPU host. These are bounded synthetic
tests, not evidence of classification accuracy or an indefinite soak test.

TLS currently reuses the verified old certificate, valid until 2026-12-24. The new
server has no ACME renewal setup yet. Configure and verify renewal as part of a
separately authorized DNS cutover; the old server's renewal setup remains unchanged.

## Server software

The user approved these installs for the replacement on 2026-11-01. A future
installation still needs approval under the working agreements. Use the configured
Ubuntu repositories for nginx and Docker. The install added five packages: `nginx`,
`nginx-common`, `docker.io`, `containerd`, and `runc`, with no removals/upgrades.
Review the refreshed simulation before proceeding. These are system-wide installs;
Docker and nginx services may start during package configuration.

```sh
apt-get update
apt-get --simulate --no-install-recommends install nginx docker.io
apt-get --no-install-recommends install nginx docker.io
docker pull tensorflow/serving@sha256:b09456481317351201fbe0b3c5954273e1b697b4a4a7834dc56f01731137a529
```

The image is TensorFlow's official Linux/amd64 2.21.0 runtime from Docker Hub, about
206 MB compressed. Image layers live in Docker's system storage. No downloaded
entrypoint script is run: the unit explicitly selects the native model-server binary.
No npm/pip packages, compiler, or native Go build dependencies are installed on the
server. This does not authorize an OS upgrade or reboot.

References: [TensorFlow Serving installation](https://www.tensorflow.org/tfx/serving/setup),
[Docker runtime](https://www.tensorflow.org/tfx/serving/docker),
[REST contract](https://www.tensorflow.org/tfx/serving/api_rest),
[2.21.0 release](https://github.com/tensorflow/serving/releases/tag/2.21.0).

## Files and service behavior

After comparing model hashes to the old deployment, install the static HTTP binary
as `/opt/hamppa/hamppa`, the upload page as `/opt/hamppa/index.html`, and the model
and label tree under `/opt/hamppa/models/`. Keep these files root-owned and readable
by the unprivileged services. Keep prior versions separately on later upgrades.

Install `hamppa-model.service` and `hamppa.service` into `/etc/systemd/system/`.
Validate the units, reload systemd, and enable/start them after image verification.
Do not replace an existing container named `hamppa-model` without inspecting its
ownership and purpose. The service creates its own ephemeral container and removes
that container when it exits; the read-only model bind mount is preserved.

The HTTP service is a systemd dynamic user with a 256 MiB memory ceiling and a
192 MiB Go memory target. The model container has a 1200 MiB memory/swap ceiling,
one CPU, and 384 PIDs. TensorFlow Serving 2.21 creates a temporary 256-thread pool
while inspecting model files, so a 128-PID limit prevents startup. The inference
thread settings still remain one each. Limits must be validated against real load on the 2 GB host.
A larger droplet and container limits alone are not evidence that the former OOM
failure is fixed. Both services restart after failure with bounded model retries.

Include `nginx-locations.conf` inside the site's server block. Preserve the existing
site root and both Game of Life aliases in that server block. Reuse the backed-up,
still-valid TLS certificate for checks with `curl --resolve`; issuing a replacement
certificate and enabling renewal require a separate cutover plan while DNS points
to the old server. Never print certificate private-key contents.

## Required checks before cutover

1. Compare all four model/label SHA-256 values with the old server and verified
   backup. Verify archive integrity and do an isolated restore/checksum pass.
2. Confirm `/readyz` returns 200 and version 2 is `AVAILABLE`. Run real JPEG and PNG
   inference, including a misleading filename. Confirm two finite probabilities
   and the original labels. Synthetic fixtures test execution, not model accuracy.
3. Exercise missing/corrupt uploads, body/file/pixel limits, and parallel requests.
   Repeat inference while recording Docker memory/PIDs/OOM state and host memory.
   Confirm the HTTP/model services remain healthy after failures and restarts.
4. Check nginx syntax, both Game of Life prefixes and assets, `/hamppa/`, and the
   inference API through the replacement IP with the intended Host header. Verify
   TLS with `--resolve` and certificate verification enabled. Test process restart
   and graceful shutdown before considering the replacement ready.
5. Confirm application ports are reachable only on loopback, model gRPC is not
   published, and existing DNS still points to the old droplet. Keep the old
   droplet running for rollback. DNS cutover is outside this deployment step.

Record measured results for each new release before declaring it ready for cutover.
