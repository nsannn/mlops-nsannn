# Security scan — training image

## Result

| Severity | Findings | Unique CVEs | Gate |
| --- | --- | --- | --- |
| CRITICAL | **0** | 0 | `trivy image --severity CRITICAL --exit-code 1` → exit 0, **PASS** |
| HIGH | 46 | 10 | Reviewed below |


## HIGH findings and decisions

| Where | Findings | CVEs | Fix available? | Decision |
| --- | --- | --- | --- | --- |
| Debian 13.7 base: util-linux family (`util-linux`, `mount`, `login`, `bsdutils`, `libmount1`, `libblkid1`, `libuuid1`, `libsmartcols1`, `liblastlog2-2`) | 36 | CVE-2026-76642, -78408, -78409, -78410 | No | Accept. Command-line system tools; the training job never calls them. Rebuild when Debian ships fixes (the base already runs `apt-get upgrade`). |
| Debian 13.7 base: ncurses (`ncurses-base`, `ncurses-bin`, `libncursesw6`, `libtinfo6`) | 4 | CVE-2025-69720 | No | Accept. Terminal library, unused by a non-interactive training container. |
| Debian 13.7 base: systemd libs (`libsystemd0`, `libudev1`) | 2 | CVE-2026-16742 | No | Accept. No init system runs in the container. |
| Debian 13.7 base: `libacl1` | 1 | CVE-2026-54369 | No | Accept. Rebuild when fixed. |
| Debian 13.7 base: `perl-base` | 1 | CVE-2026-9538 | No | Accept. No Perl code runs in the image. |
| Python: `cryptography` 49.0.0 (PKCS#7 EnvelopedData decryption) | 1 | CVE-2026-69247 | 50.0.0, **blocked**: `mlflow==3.15.2` requires `cryptography<50` | Accept. Training never decrypts PKCS#7 data. Revisit when the course moves past mlflow 3.15.2. |
| Python: `mlflow` 3.15.2 (SSRF via AI Gateway secret) | 1 | CVE-2026-71211 | No | Accept. The flaw is in the MLflow **server**'s AI Gateway; this image only runs the tracking client. |

