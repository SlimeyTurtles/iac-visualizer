# Auto-deploy to EC2 (one-time setup)

After this is wired up, every push to `main` deploys to your EC2 instance. The workflow lives at [.github/workflows/deploy.yml](../.github/workflows/deploy.yml).

## Architecture

GitHub Actions (free shared runner) → SSH into EC2 → `git pull` + `npm ci` + PM2 restart. No daemon on the EC2 box other than PM2.

## One-time EC2 setup

SSH into the instance and do these steps once.

### 1. Install Node.js + PM2

```bash
# Node 20 LTS (skip if already installed)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# PM2 globally
sudo npm install -g pm2
```

### 2. Clone the repo

```bash
cd ~                 # or wherever you want it
git clone https://github.com/SlimeyTurtles/iac-visualizer.git
cd iac-visualizer
npm ci --omit=dev
```

Note the absolute path (e.g. `/home/ubuntu/iac-visualizer`) — you'll need it for the GitHub secret `EC2_PROJECT_PATH`.

### 3. Start the app once + enable boot persistence

```bash
pm2 start npm --name iac-visualizer -- start
pm2 save

# Make PM2 start on boot. This prints a one-line `sudo` command — run it as-is.
pm2 startup
```

Test that `http://<ec2-public-ip>:3000/` responds. Open port 3000 in the EC2 security group if not already.

### 4. Add a dedicated deploy SSH key

Don't reuse your personal key. Generate a fresh ed25519 keypair on your local machine:

```bash
ssh-keygen -t ed25519 -C "github-deploy@iac-visualizer" -f ~/.ssh/iac_deploy
```

Copy the **public** key onto the EC2 box (append to authorized_keys):

```bash
cat ~/.ssh/iac_deploy.pub | ssh ubuntu@<ec2-host> 'cat >> ~/.ssh/authorized_keys'
```

Verify you can SSH using just this key:

```bash
ssh -i ~/.ssh/iac_deploy ubuntu@<ec2-host> 'echo ok'
```

## GitHub repo secrets

In <https://github.com/SlimeyTurtles/iac-visualizer/settings/secrets/actions>, add **Secrets**:

| Name | Value |
|---|---|
| `EC2_HOST` | The EC2 public DNS or IP, e.g. `ec2-12-34-56-78.compute-1.amazonaws.com` |
| `EC2_USER` | `ubuntu` (Ubuntu AMI) or `ec2-user` (Amazon Linux) |
| `EC2_SSH_KEY` | The **full contents** of `~/.ssh/iac_deploy` (the private key, including `-----BEGIN…-----` and `-----END…-----` lines) |
| `EC2_PROJECT_PATH` | Absolute path on EC2, e.g. `/home/ubuntu/iac-visualizer` |
| `EC2_PORT` | *(optional)* SSH port if not 22 |

And optionally a **Variable** (not secret — same Settings page, "Variables" tab):

| Name | Value |
|---|---|
| `PM2_APP_NAME` | *(optional)* Defaults to `iac-visualizer` if not set |

## Verify

Push any commit to `main` and watch the Actions tab. The job should complete in ~30 seconds. If it fails, the SSH-action output shows the exact remote command output — read that first before re-running.

To deploy a hotfix without pushing, you can use the **"Run workflow"** button on the Actions tab (this is what `workflow_dispatch` enables).

## Troubleshooting

- **`Permission denied (publickey)`**: the private key in `EC2_SSH_KEY` doesn't match anything in EC2's `~/.ssh/authorized_keys`, or `EC2_USER` is wrong. Re-run step 4.
- **`pm2: command not found`**: PM2 wasn't installed globally or `~/.npm-global/bin` isn't in the runner's PATH. The workflow uses `pm2` bare; if it's installed but not on PATH, edit the workflow to call it by absolute path (e.g. `/usr/local/bin/pm2`).
- **Workflow succeeds but site is still old**: the server is cached client-side. `server.js` already sets `Cache-Control: no-store`, but check whether anything (CloudFront, nginx) sits in front and caches.
- **`npm ci` fails**: the `package-lock.json` on EC2 may be stale or missing. The workflow's `git reset --hard origin/main` overwrites it with the committed version, so this only happens if the lockfile is missing from the repo entirely.

## Rolling back

```bash
# On the EC2 box
cd /home/ubuntu/iac-visualizer
git log --oneline -10           # find the commit you want
git reset --hard <commit-sha>
pm2 restart iac-visualizer
```

The next push to `main` will overwrite this — that's the point — so use rollback as a bridge to fixing the broken commit.
