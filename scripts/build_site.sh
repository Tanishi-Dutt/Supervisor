#!/bin/sh
# Builds the static website into public/ (Vercel runs this; run it locally to preview).
# FR_API = backend base URL. Empty (the default) means same origin: Vercel proxies /api/* to Render.
set -e
cd "$(dirname "$0")/.."
rm -rf public && mkdir -p public
cp -r site/* public/
printf 'window.FR_API = "%s";\n' "${FR_API:-}" > public/config.js
sed 's#<!--FR_CONFIG-->#<script src="config.js"></script>#' viewer/index.html > public/city.html
echo "built public/ (FR_API='${FR_API:-}')"
