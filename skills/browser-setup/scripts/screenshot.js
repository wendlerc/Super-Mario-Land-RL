const puppeteer = require('puppeteer');

(async () => {
  try {
    console.log('Launching Chrome with custom libraries...');
    const browser = await puppeteer.launch({
      headless: true,
      executablePath: '/data/workspace/chrome-install/opt/google/chrome/google-chrome',
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
    });
    console.log('Browser launched successfully!');
    
    const page = await browser.newPage();
    await page.goto('https://people.epfl.ch/robert.west', { waitUntil: 'networkidle0' });
    console.log('Page loaded');
    
    await page.screenshot({ path: '/data/workspace/robert_west.png', fullPage: true });
    console.log('Screenshot saved to /data/workspace/robert_west.png');
    
    await browser.close();
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
})();
