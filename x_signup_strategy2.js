const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');

puppeteer.use(StealthPlugin());

(async () => {
  try {
    console.log('Launching stealth browser (Strategy #2 - Mobile UA)...');
    const browser = await puppeteer.launch({
      headless: true,
      executablePath: '/data/workspace/chrome-install/opt/google/chrome/google-chrome',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage'
      ]
    });
    
    const page = await browser.newPage();
    
    // Strategy #2: Mobile user agent + longer delays
    await page.setUserAgent(
      'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1'
    );
    await page.setViewport({ width: 390, height: 844 }); // iPhone viewport
    
    // Navigate to X signup with longer timeout
    console.log('Navigating to X signup (mobile)...');
    await page.goto('https://x.com/i/flow/signup', { 
      waitUntil: 'domcontentloaded',
      timeout: 90000 
    });
    
    // Wait for page to fully load
    console.log('Waiting for content to load...');
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    console.log('Taking screenshot...');
    await page.screenshot({ path: '/data/workspace/x_signup_attempt_2.png', fullPage: true });
    console.log('Screenshot saved to x_signup_attempt_2.png');
    
    await browser.close();
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
})();
