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
    console.log('Filling DOB...');
    await page.type('input[data-testid="ocfEnterTextDateInput"]', '01/01/2000', { delay: 50 });
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Click Next
    console.log('Clicking "Next"...');
    await page.evaluate(() => {
      const xpath = "//div[@role='button']//span[text()='Next']";
      const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
      if (result.singleNodeValue) {
        result.singleNodeValue.closest('div[role="button"]').click();
      }
    });
    await new Promise(resolve => setTimeout(resolve, 8000));
    
    // Take screenshot to see what happened
    console.log('Taking screenshot after clicking Next...');
    await page.screenshot({ path: '/data/workspace/x_signup_after_next.png', fullPage: true });
    
    // Check page content
    const html = await page.content();
    console.log('Page HTML length:', html.length);
    
    // Check for verification code field
    const hasVerificationInput = await page.$('input[inputmode="numeric"]') !== null || 
                                  await page.$('input[placeholder*="code" i]') !== null ||
                                  await page.$('input[name="verification_code"]') !== null;
    
    console.log('Has verification code input:', hasVerificationInput);
    
    // Check for phone verification
    const hasPhoneInput = await page.$('input[type="tel"]') !== null ||
                          await page.$('input[name="phone_number"]') !== null;
    
    console.log('Has phone input:', hasPhoneInput);
    
    // Check page text for keywords
    const pageText = await page.evaluate(() => document.body.innerText);
    console.log('Page contains "verification":', pageText.toLowerCase().includes('verification'));
    console.log('Page contains "phone":', pageText.toLowerCase().includes('phone'));
    console.log('Page contains "confirm":', pageText.toLowerCase().includes('confirm'));
    
    console.log('Screenshot saved to x_signup_after_next.png');
    
    await browser.close();
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
})();
