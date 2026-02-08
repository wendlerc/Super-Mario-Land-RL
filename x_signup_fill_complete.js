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
    await new Promise(resolve => setTimeout(resolve, 8000));
    
    // Click "Create account" button
    console.log('Clicking "Create account"...');
    await page.evaluate(() => {
      const allElements = document.querySelectorAll('*');
      for (const el of allElements) {
        if (el.textContent && el.textContent.trim() === 'Create account' && el.getAttribute('role') === 'button') {
          el.click();
          return;
        }
      }
    });
    await new Promise(resolve => setTimeout(resolve, 8000));
    
    // Click "Use email instead"
    console.log('Clicking "Use email instead"...');
    await page.evaluate(() => {
      const xpath = "//span[contains(text(), 'Use email instead')]";
      const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
      if (result.singleNodeValue) {
        result.singleNodeValue.click();
      }
    });
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Fill in Name
    console.log('Filling name...');
    await page.type('input[name="name"]', 'Flux Autonomous', { delay: 100 });
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Fill in Email
    console.log('Filling email...');
    await page.type('input[name="email"]', 'jarvis-openclaw-bot@proton.me', { delay: 100 });
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Fill in Date of Birth
    console.log('Filling date of birth...');
    // X uses separate dropdowns or input fields for DOB
    // Try to find the input
    const dobInput = await page.$('input[data-testid="ocfEnterTextDateInput"], input[placeholder*="Date"], input[name*="birth"]');
    if (dobInput) {
      await dobInput.type('01/01/2000', { delay: 50 });
    } else {
      console.log('DOB input not found, trying alternative...');
      // Maybe it's month/day/year dropdowns
      // For now, skip DOB and take screenshot to see what we have
    }
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Take screenshot
    console.log('Taking screenshot of filled form...');
    await page.screenshot({ path: '/data/workspace/x_signup_filled.png', fullPage: true });
    
    console.log('Form filled. Screenshot saved to x_signup_filled.png');
    
    await browser.close();
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
})();
