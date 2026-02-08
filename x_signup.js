const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');

puppeteer.use(StealthPlugin());

(async () => {
  try {
    console.log('Launching stealth browser...');
    const browser = await puppeteer.launch({
      headless: true,
      executablePath: '/data/workspace/chrome-install/opt/google/chrome/google-chrome',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--no-first-run',
        '--no-zygote',
        '--disable-gpu'
      ]
    });
    
    const page = await browser.newPage();
    
    // Set realistic viewport
    await page.setViewport({ width: 1920, height: 1080 });
    
    // Navigate to X signup
    console.log('Navigating to X signup...');
    await page.goto('https://x.com/i/flow/signup', { 
      waitUntil: 'networkidle0',
      timeout: 60000 
    });
    
    console.log('Page loaded, taking screenshot...');
    await page.screenshot({ path: '/data/workspace/x_signup_attempt_1.png', fullPage: true });
    console.log('Screenshot saved to x_signup_attempt_1.png');
    
    await browser.close();
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
})();
