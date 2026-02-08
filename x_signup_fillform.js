const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');

puppeteer.use(StealthPlugin());

(async () => {
  try {
    console.log('Launching browser for X signup...');
    const browser = await puppeteer.launch({
      headless: true,
      executablePath: '/data/workspace/chrome-install/opt/google/chrome/google-chrome',
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
    });
    
    const page = await browser.newPage();
    
    // Mobile user agent
    await page.setUserAgent(
      'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1'
    );
    await page.setViewport({ width: 390, height: 844 });
    
    // Go to signup
    console.log('Loading X signup...');
    await page.goto('https://x.com/i/flow/signup', { 
      waitUntil: 'domcontentloaded',
      timeout: 90000 
    });
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Click "Create account" button using JavaScript evaluation
    console.log('Clicking "Create account"...');
    await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('div[role="button"]'));
      const createAccountBtn = buttons.find(btn => btn.textContent.includes('Create account'));
      if (createAccountBtn) createAccountBtn.click();
    });
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Take screenshot of the form
    console.log('Taking screenshot...');
    await page.screenshot({ path: '/data/workspace/x_signup_step1.png', fullPage: true });
    
    // Now fill in the form
    console.log('Filling signup form...');
    
    // Name field
    await page.type('input[name="name"]', 'Flux Autonomous', { delay: 100 });
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Email field - use my proton email
    await page.type('input[name="email"]', 'jarvis-openclaw-bot@proton.me', { delay: 100 });
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Take screenshot after filling
    console.log('Form filled, taking screenshot...');
    await page.screenshot({ path: '/data/workspace/x_signup_step2.png', fullPage: true });
    
    console.log('Screenshots saved. Check them for next steps.');
    
    await browser.close();
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
})();
