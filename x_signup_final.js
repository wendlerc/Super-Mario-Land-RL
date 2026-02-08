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
    
    // Fill in Name with proper events
    console.log('Filling name...');
    await page.evaluate(() => {
      const input = document.querySelector('input[name="name"]');
      if (input) {
        input.value = 'Flux Autonomous';
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Fill in Email with proper events
    console.log('Filling email...');
    await page.evaluate(() => {
      const input = document.querySelector('input[name="email"]');
      if (input) {
        input.value = 'jarvis-openclaw-bot@proton.me';
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Fill in DOB with proper events
    console.log('Filling DOB...');
    await page.evaluate(() => {
      const input = document.querySelector('input[type="date"], input[data-testid*="Date"]');
      if (input) {
        input.value = '2000-01-01'; // YYYY-MM-DD format for date inputs
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Take screenshot before clicking Next
    await page.screenshot({ path: '/data/workspace/x_before_next.png', fullPage: true });
    
    // Click Next with proper event simulation
    console.log('Clicking "Next"...');
    await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('div[role="button"]'));
      const nextBtn = buttons.find(btn => btn.textContent && btn.textContent.trim() === 'Next');
      if (nextBtn) {
        // Dispatch multiple events to ensure click registers
        nextBtn.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
        nextBtn.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
        nextBtn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        nextBtn.click();
      }
    });
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    // Take screenshot to see what happened
    console.log('Taking screenshot after clicking Next...');
    await page.screenshot({ path: '/data/workspace/x_after_next_v2.png', fullPage: true });
    
    // Get page info
    const pageText = await page.evaluate(() => document.body.innerText);
    console.log('Page text (first 500 chars):', pageText.substring(0, 500));
    
    await browser.close();
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
})();
