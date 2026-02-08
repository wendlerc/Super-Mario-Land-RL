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
    
    // Find and click "Create account" button by text content
    console.log('Clicking "Create account"...');
    const clicked = await page.evaluate(() => {
      const allElements = document.querySelectorAll('*');
      for (const el of allElements) {
        if (el.textContent && el.textContent.trim() === 'Create account' && el.getAttribute('role') === 'button') {
          el.click();
          return true;
        }
      }
      return false;
    });
    
    if (!clicked) {
      console.log('Button not found via JS, trying xpath...');
      await page.evaluate(() => {
        const xpath = "//div[@role='button'][contains(text(), 'Create account')]";
        const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
        if (result.singleNodeValue) {
          result.singleNodeValue.click();
        }
      });
    }
    
    await new Promise(resolve => setTimeout(resolve, 8000));
    
    // Take screenshot of what we see now
    console.log('Taking screenshot...');
    await page.screenshot({ path: '/data/workspace/x_signup_after_click.png', fullPage: true });
    
    // Get page HTML to see what's there
    const html = await page.content();
    console.log('Page HTML length:', html.length);
    
    // Check if we're on the form now
    const hasNameInput = await page.$('input[name="name"]') !== null;
    const hasPhoneInput = await page.$('input[type="tel"]') !== null;
    
    console.log('Has name input:', hasNameInput);
    console.log('Has phone input:', hasPhoneInput);
    
    console.log('Screenshot saved. Check x_signup_after_click.png');
    
    await browser.close();
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
})();
