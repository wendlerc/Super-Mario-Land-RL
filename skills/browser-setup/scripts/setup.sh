#!/bin/bash
# Browser Setup Script for Flux
# Run this to restore browser functionality after system reset

echo "=== Browser Setup Script ==="
echo "This will download and configure Chrome in /data/workspace"
echo ""

# Create directories
mkdir -p /data/workspace/chrome-install
mkdir -p /data/workspace/browser-libs
mkdir -p /data/workspace/browser-libs/usr/lib/x86_64-linux-gnu

cd /data/workspace/browser-libs

echo "Downloading required libraries..."

# Core X11 libraries
echo "  - X11 libraries..."
wget -q http://http.us.debian.org/debian/pool/main/libx/libx11-xcb/libx11-xcb1_1.8.4-2+deb12u2_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/libx/libxcomposite/libxcomposite1_0.4.5-1_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/libx/libxcursor/libxcursor1_1.2.1-1_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/libx/libxdamage/libxdamage1_1.1.6-1_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/libx/libxfixes/libxfixes3_6.0.0-2_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/libx/libxrandr/libxrandr2_1.5.2-2+b1_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/libx/libxi/libxi6_1.8-1+b1_amd64.deb

# Security/Network libraries
echo "  - Security libraries..."
wget -q http://http.us.debian.org/debian/pool/main/n/nspr/libnspr4_4.35-1_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/n/nss/libnss3_3.87.1-1+deb12u1_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/d/dbus/libdbus-1-3_1.14.10-1~deb12u1_amd64.deb

# Accessibility libraries
echo "  - Accessibility libraries..."
wget -q http://http.us.debian.org/debian/pool/main/a/atk1.0/libatk1.0-0_2.36.0-2_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/a/at-spi2-core/libatk-bridge2.0-0_2.46.0-5_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/a/at-spi2-core/libatspi2.0-0_2.46.0-5_amd64.deb

# Graphics/Audio libraries
echo "  - Graphics/Audio libraries..."
wget -q http://http.us.debian.org/debian/pool/main/m/mesa/libgbm1_22.3.6-1+deb12u1_amd64.deb
wget -q http://archive.ubuntu.com/ubuntu/pool/main/libx/libxkbcommon/libxkbcommon0_1.4.0-1_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/a/alsa-lib/libasound2_1.2.8-1+b1_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/libd/libdrm/libdrm2_2.4.114-1+b1_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/c/cups/libcups2_2.4.2-3+deb12u9_amd64.deb

# Network discovery
echo "  - Network libraries..."
wget -q http://archive.ubuntu.com/ubuntu/pool/main/a/avahi/libavahi-common3_0.8-5ubuntu5_amd64.deb
wget -q http://archive.ubuntu.com/ubuntu/pool/main/a/avahi/libavahi-client3_0.8-5ubuntu5_amd64.deb

# Wayland libraries
echo "  - Wayland libraries..."
wget -q http://http.us.debian.org/debian/pool/main/w/wayland/libwayland-server0_1.21.0-1_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/w/wayland/libwayland-client0_1.21.0-1_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/w/wayland/libwayland-cursor0_1.21.0-1_amd64.deb
wget -q http://http.us.debian.org/debian/pool/main/w/wayland/libwayland-egl1_1.21.0-1_amd64.deb

# Extract all .deb packages
echo "Extracting libraries..."
for deb in *.deb; do
  dpkg -x "$deb" . 2>/dev/null
  rm "$deb" 2>/dev/null
done

# Copy from lib/ to usr/lib/ if needed
if [ -d "lib/x86_64-linux-gnu" ]; then
  cp -r lib/x86_64-linux-gnu/* usr/lib/x86_64-linux-gnu/ 2>/dev/null
fi

echo ""
echo "Downloading Chrome..."
cd /data/workspace
curl -L -o google-chrome-stable_current_amd64.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
dpkg -x google-chrome-stable_current_amd64.deb chrome-install
rm google-chrome-stable_current_amd64.deb

echo ""
echo "Installing Playwright..."
npm init -y 2>/dev/null
npm install playwright puppeteer

echo ""
echo "=== Setup Complete ==="
echo "Chrome: /data/workspace/chrome-install/opt/google/chrome/"
echo "Libraries: /data/workspace/browser-libs/usr/lib/x86_64-linux-gnu/"
echo ""
echo "To use Chrome:"
echo "  export LD_LIBRARY_PATH=/data/workspace/browser-libs/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
echo "  /data/workspace/chrome-install/opt/google/chrome/google-chrome --version"
echo ""
echo "Test screenshot:"
echo "  node /data/workspace/screenshot.js"
