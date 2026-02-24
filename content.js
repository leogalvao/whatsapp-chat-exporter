// WhatsApp Chat Exporter - Content Script (v1.4 - date/time reliability fixes)
// Interacts with WhatsApp Web to extract chat data

(function() {
  'use strict';

  // BUG-01 fix: Read the guard flag but do NOT return early.  When Chrome
  // invalidates the content-script context (extension update, etc.) the old
  // listener is dead but window.__whatsappExporterLoaded persists.  Re-injection
  // must always re-register the listener.  The flag is still used to skip
  // one-time initialisation code further down.
  const _alreadyLoaded = window.__whatsappExporterLoaded;
  window.__whatsappExporterLoaded = true;

  // Debug mode - set to false to suppress console logs
  const DEBUG = true;

  function log(...args) {
    if (DEBUG) console.log('[WA Exporter]', ...args);
  }

  // -----------------------------------------------------------------------
  // IMP-08: Centralised selector registry — one place to update when
  // WhatsApp changes its DOM.
  // -----------------------------------------------------------------------
  const SELECTORS = {
    chatList: [
      '[data-testid="cell-frame-container"]',
      '[data-testid="list-item-link"]',
      '[aria-label="Chat list"] [role="listitem"]',
      '#pane-side [role="listitem"]',
      '#pane-side [role="row"]',
    ],
    chatListFallback: '#pane-side [tabindex="-1"]',
    chatName: ['[title]', 'span[dir="auto"]', 'span[title]'],
    scrollContainer: [
      '[data-testid="conversation-panel-body"]',
      '#main [role="application"]',
      '#main .copyable-area > div',
    ],
    dateDivider: [
      '[data-testid="date-divider"]',
      '[data-testid="msg-date-divider"]',
    ],
    dateDividerWildcard: '[data-testid*="date"]',
    msgContainer: '[data-testid="msg-container"]',
    msgMeta: '[data-testid="msg-meta"]',
    msgText: '[data-testid="msg-text"]',
    author: [
      '[data-testid="msg-author-title"]',
      '[data-testid="author"]',
      'span[aria-label*="@"]',
      '[data-testid="msg-author"]',
      '[data-testid="msg-author-name"]',
    ],
    image: 'img[src*="blob:"], img[src*="media"]',
    video: '[data-testid*="video"], video',
    audio: '[data-testid*="ptt"], [data-testid*="audio"]',
    document: '[data-testid*="document"]',
    systemMsg: '[data-testid="msg-system"]',
    reactions: '[data-testid="reactions"]',
    forwarded: '[data-testid*="forwarded"], [data-testid="forward-refreshed"]',
    deleted: '[data-icon="recalled"]',
    outgoing: '[data-icon="msg-dblcheck"], [data-icon="msg-check"], [data-testid="msg-dblcheck"], [data-testid="msg-check"]',
    quotedMessage: '[data-testid="quoted-message"]',
    quotedAuthor: '[data-testid="quoted-message-author"]',
    quotedText: '[data-testid="quoted-message-text"]',
    header: '#main header',
    sidePanel: '#pane-side',
    mainPanel: '#main',
    conversationBody: '[data-testid="conversation-panel-body"]',
    conversationBodyFallback: '[role="application"]',
  };

  // BUG-02: Selector probe — validates which selectors actually match the
  // live DOM.  Logs results and returns the first working chat-list selector.
  function selectorProbe() {
    const results = {};
    let chatListSelector = null;
    for (const sel of SELECTORS.chatList) {
      const count = document.querySelectorAll(sel).length;
      results[sel] = count;
      if (count > 0 && !chatListSelector) chatListSelector = sel;
    }
    const mainPanel = document.querySelector(SELECTORS.mainPanel);
    results['#main'] = mainPanel ? 'found' : 'NOT FOUND';
    log('Selector probe:', results);
    return { results, chatListSelector, healthy: chatListSelector !== null };
  }

  // IMP-02: Global abort controller for the current export
  let currentAbortController = null;

  // BUG-07: Detected date-part order ('mdy' or 'dmy') — set once at load
  let detectedDateOrder = null;

  function detectDateLocale() {
    const lang = (document.documentElement?.lang || '').toLowerCase();
    if (lang.startsWith('en-us') || lang === 'en') return 'mdy';
    const dmyPrefixes = [
      'pt','es','fr','de','it','nl','ru','ar','hi','tr','pl','uk','ro',
      'el','cs','sv','da','fi','no','hu','bg','hr','sk','sl','sr','lt','lv','et'
    ];
    for (const p of dmyPrefixes) { if (lang.startsWith(p)) return 'dmy'; }
    // Fallback: sniff UI strings
    const sample = (document.body?.textContent || '').substring(0, 5000);
    if (/Escribe un mensaje|Escreva uma mensagem|Écrivez un message|Nachricht eingeben|Scrivi un messaggio/.test(sample)) return 'dmy';
    if (sample.includes('Type a message')) return 'mdy';
    return null;
  }

  // --- One-time initialisation (skipped on re-injection) ---
  if (!_alreadyLoaded) {
    detectedDateOrder = detectDateLocale();
    log('Detected date locale order:', detectedDateOrder || 'unknown');
    setTimeout(() => selectorProbe(), 3000);
  } else {
    log('Re-injection detected — re-registering message listener');
    detectedDateOrder = detectDateLocale();
  }

  // Store for extracted data
  let exportData = {
    chatName: '',
    exportDate: '',
    messages: [],
    systemMessages: [],
    media: []
  };

  // Listen for messages from popup
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    handleMessage(message, sendResponse);
    return true;
  });

  async function handleMessage(message, sendResponse) {
    try {
      switch (message.action) {
        // IMP-01: Health-check ping
        case 'ping':
          sendResponse({ pong: true, version: '1.5' });
          break;

        case 'getChats': {
          const chats = await getChatList();
          if (chats.length === 0) {
            // BUG-02: Surface clear error when zero selectors match
            const probe = selectorProbe();
            if (!probe.healthy) {
              sendResponse({
                success: false,
                error: 'DOM_CHANGED',
                detail: 'No chat-list selectors matched the current WhatsApp Web DOM. The page structure may have changed.'
              });
              return;
            }
          }
          sendResponse({ success: true, chats: chats });
          break;
        }

        case 'exportChat': {
          // IMP-02: Create abort controller for this export
          currentAbortController = new AbortController();
          const opts = message.options || {};
          opts._signal = currentAbortController.signal;
          // BUG-10: Carry session ID for progress filtering
          if (message.sessionId) opts._sessionId = message.sessionId;
          try {
            const result = await exportChat(opts);
            sendResponse(result);
          } finally {
            currentAbortController = null;
          }
          break;
        }

        // IMP-02: Cancel a running export
        case 'cancelExport':
          if (currentAbortController) {
            currentAbortController.abort();
            sendResponse({ success: true });
          } else {
            sendResponse({ success: false, error: 'No export in progress' });
          }
          break;

        default:
          sendResponse({ success: false, error: 'Unknown action' });
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        log('Export cancelled by user');
        sendResponse({ success: false, error: 'Export cancelled', cancelled: true });
      } else {
        console.error('WhatsApp Exporter Error:', error);
        sendResponse({ success: false, error: error.message });
      }
    }
  }

  // Get list of available chats from sidebar
  async function getChatList() {
    const chats = [];
    const maxRetries = 10; // 10 * 500ms = 5 seconds
    let attempts = 0;

    log('getChatList: Starting scan...');

    while (attempts < maxRetries) {
      // Try multiple selector strategies (IMP-08: from central registry)
      const selectors = SELECTORS.chatList;

      let chatElements = [];

      for (const selector of selectors) {
        chatElements = document.querySelectorAll(selector);
        if (chatElements.length > 0) {
          log(`Found ${chatElements.length} chats with selector: ${selector}`);
          break;
        }
      }

      if (chatElements.length === 0) {
        const sidePanel = document.querySelector(SELECTORS.sidePanel);
        if (sidePanel) {
          chatElements = sidePanel.querySelectorAll(SELECTORS.chatListFallback);
          if (chatElements.length > 0) {
            log(`Fallback: found ${chatElements.length} potential chats via tabindex`);
          }
        }
      }

      if (chatElements.length > 0) {
        // Process found elements
        chatElements.forEach((element, index) => {
          let name = '';

          const titleEl = element.querySelector('[title]');
          if (titleEl) {
            name = titleEl.getAttribute('title');
          }

          if (!name) {
            const spanEl = element.querySelector('span[dir="auto"], span[title]');
            if (spanEl) {
              name = spanEl.getAttribute('title') || spanEl.textContent;
            }
          }

          name = (name || '').trim();

          if (name && name.length > 0) {
            chats.push({
              id: index,
              name: name
            });
          }
        });
        
        // If we successfully extracted names, break the retry loop
        if (chats.length > 0) break;
      }

      // Wait and retry
      attempts++;
      if (attempts < maxRetries) {
        log(`getChatList: Attempt ${attempts} found 0 chats. Retrying...`);
        await sleep(500);
      }
    }

    const uniqueChats = chats.filter((chat, index, self) =>
      index === self.findIndex(c => c.name === chat.name)
    );

    log(`Final chat list: ${uniqueChats.length} unique chats found`);
    return uniqueChats;
  }

  // Export selected chat
  async function exportChat(options) {
    // BUG-10: Track session ID for progress filtering
    _currentSessionId = options._sessionId || null;

    exportData = {
      chatName: options.chatName,
      exportDate: new Date().toISOString(),
      options: options,
      messages: [],
      systemMessages: [],
      media: []
    };

    sendProgress(5, 'Opening chat...', 'Navigating to selected chat');

    const chatOpened = await openChat(options.chatIndex, options.chatName);
    if (!chatOpened) {
      throw new Error('Could not open the selected chat');
    }

    sendProgress(10, 'Loading messages...', 'Waiting for messages to load');

    // Wait for the chat to actually render content before proceeding.
    // WhatsApp may take several seconds to load messages after clicking a chat.
    const chatReady = await waitForChatContent(8000);
    if (!chatReady) {
      log('WARNING: Chat content did not appear within timeout — proceeding anyway');
    }

    // Determine scroll depth from options
    // Higher counts let us reach the very beginning of large chats;
    // early-exit logic (unchanged scroll height) stops it sooner in practice.
    const scrollDepthMap = { 'recent': 30, '500': 100, 'full': 500 };
    const maxScrolls = scrollDepthMap[options.scrollDepth] || 30;

    sendProgress(15, 'Loading history...', 'Scrolling to load messages');
    const scrollContainer = await scrollToLoadMessages(maxScrolls, options._signal);

    // Combined scroll-forward + extraction pass.
    // Scrolls through the entire chat from top to bottom, extracting messages
    // at each viewport position.  This ensures all messages are captured even
    // when WhatsApp virtualises its message list, and also triggers
    // lazy-loading of media elements (images, etc.).
    sendProgress(28, 'Extracting messages...', 'Scanning chat content');
    let result = await scrollAndExtractMessages(scrollContainer, options);

    // Retry once if extraction returned 0 messages — the DOM may not have
    // been fully ready on the first pass.
    if (result.messages.length === 0) {
      log('WARNING: 0 messages found — retrying extraction after 3 s wait');
      sendProgress(30, 'Retrying extraction...', 'First pass found 0 messages, retrying');
      await sleep(3000);
      const retryContainer = await scrollToLoadMessages(Math.min(maxScrolls, 30), options._signal);
      result = await scrollAndExtractMessages(retryContainer || scrollContainer, options);
      log(`Retry result: ${result.messages.length} messages`);
    }

    exportData.messages = result.messages;
    exportData.systemMessages = result.systemMessages;

    log(`Extracted ${result.messages.length} messages, ${result.systemMessages.length} system messages`);

    sendProgress(80, 'Processing media...', `Found ${result.messages.length} messages`);

    if (options.includeMedia && result.messages.length > 0) {
      await extractMedia(result.messages);
    }

    // IMP-07: Validate export before download
    const validation = validateExport(result.messages);
    if (!validation.valid) {
      log('Export validation warnings:', validation.warnings);
    }

    sendProgress(90, 'Creating export...', 'Generating export file');

    if (options.returnData) {
      // Return file data to the caller (popup) instead of downloading.
      // The popup writes files directly to the user-chosen directory.
      const files = buildExportFiles(options.chatName, options.exportFormat || 'json');
      sendProgress(100, 'Export complete!', `Exported ${result.messages.length} messages`);
      return {
        success: true,
        messageCount: result.messages.length,
        mediaCount: exportData.media.length,
        files: files,
        warnings: validation.warnings
      };
    }

    await downloadExport(options.chatName, options.exportFormat || 'json');

    sendProgress(100, 'Export complete!', `Exported ${result.messages.length} messages`);

    return {
      success: true,
      messageCount: result.messages.length,
      mediaCount: exportData.media.length,
      warnings: validation.warnings
    };
  }

  // Scroll to load more messages — accepts configurable max iterations.
  // Returns the scroll container so callers can reuse it.
  async function scrollToLoadMessages(maxIterations, signal) {
    const scrollSelectors = [
      '[data-testid="conversation-panel-body"]',
      '#main [role="application"]',
      '#main .copyable-area > div',
    ];

    // Retry finding the scroll container — in batch mode the chat may still
    // be loading when we arrive here.  Poll for up to 10 seconds.
    let scrollContainer = null;
    for (let attempt = 0; attempt < 20; attempt++) {
      for (const selector of scrollSelectors) {
        scrollContainer = document.querySelector(selector);
        if (scrollContainer && scrollContainer.scrollHeight > 0) {
          log(`Found scroll container: ${selector} (attempt ${attempt + 1})`);
          break;
        }
      }
      if (scrollContainer) break;
      await sleep(500);
    }

    // Fallback: find the scrollable ancestor of message elements.
    // WhatsApp may have removed data-testid and role attributes from the
    // scroll container, so walk up from a known message element.
    if (!scrollContainer) {
      log('Scroll container not found via selectors, trying ancestor walk');
      const msgEl = document.querySelector('#main [data-id]');
      if (msgEl) {
        let parent = msgEl.parentElement;
        while (parent && parent !== document.body && parent.id !== 'main') {
          const style = getComputedStyle(parent);
          if ((style.overflowY === 'auto' || style.overflowY === 'scroll')
              && parent.scrollHeight > parent.clientHeight) {
            scrollContainer = parent;
            log(`Found scroll container via ancestor walk: tag=${parent.tagName}, class=${parent.className?.substring(0, 60)}`);
            break;
          }
          parent = parent.parentElement;
        }
      }
    }

    if (!scrollContainer) {
      log('ERROR: Could not find scroll container after 10 s of retries');
      return null;
    }

    let unchangedCount = 0;
    let prevScrollHeight = scrollContainer.scrollHeight;

    for (let i = 0; i < maxIterations; i++) {
      scrollContainer.scrollTop = 0;
      // IMP-06: adaptive delay — resolves early when DOM settles, max 3 s
      await adaptiveSleep(3000, signal);

      const newScrollHeight = scrollContainer.scrollHeight;
      if (newScrollHeight === prevScrollHeight) {
        unchangedCount++;
        // Require 5 consecutive unchanged heights before giving up, so transient
        // network delays don't truncate the export prematurely.
        if (unchangedCount >= 5) {
          log(`Scroll stopped early at iteration ${i + 1} — no new content loaded`);
          break;
        }
      } else {
        unchangedCount = 0;
      }
      prevScrollHeight = newScrollHeight;

      sendProgress(
        15 + (i / maxIterations * 13),
        'Loading history...',
        `Scroll ${i + 1} of ${maxIterations}`
      );
    }

    // Leave scroll position at top so the forward media pass starts from there.
    scrollContainer.scrollTop = 0;
    await sleep(500);

    return scrollContainer;
  }

  // Wait until the chat panel has at least one message element or a date
  // divider, indicating that WhatsApp has finished loading content.
  // Returns true if content appeared, false on timeout.
  async function waitForChatContent(timeoutMs) {
    const mainPanel = document.querySelector('#main');
    if (!mainPanel) return false;

    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      // Check for any sign of message content
      if (mainPanel.querySelector('[data-id]') ||
          mainPanel.querySelector('[data-testid="msg-container"]') ||
          mainPanel.querySelector('[data-testid="date-divider"]') ||
          mainPanel.querySelector('[data-testid="msg-date-divider"]') ||
          mainPanel.querySelector('[role="row"]')) {
        log('Chat content detected');
        return true;
      }
      await sleep(500);
    }
    return false;
  }

  // -----------------------------------------------------------------------
  // Scroll from top to bottom, extracting messages at each viewport position.
  // This replaces the old separate "scroll forward for media" + "extract
  // messages" approach.  By reading the DOM at every scroll stop we capture
  // ALL messages even when WhatsApp virtualises its list, and we also pick
  // up date-divider elements as they scroll into view.
  // -----------------------------------------------------------------------
  async function scrollAndExtractMessages(scrollContainer, options) {
    const mainPanel = document.querySelector('#main');
    if (!mainPanel) {
      log('ERROR: #main panel not found');
      return { messages: [], systemMessages: [] };
    }

    const seenIds = new Set();
    const seenHashes = new Set(); // IMP-03: secondary dedup by content hash
    const messages = [];
    const systemMessages = [];
    let lastKnownDateString = null;

    // Helper: find message elements using multiple strategies (same logic
    // that was previously inside extractMessages).
    function findMessageElements() {
      // Strategy 1: data-id (most reliable)
      let els = mainPanel.querySelectorAll('[data-id]');
      if (els.length > 0) return Array.from(els);

      // Strategy 2: msg-container → walk up to row
      els = mainPanel.querySelectorAll('[data-testid="msg-container"]');
      if (els.length > 0) {
        const rows = new Set();
        els.forEach(el => {
          const row = el.closest('[role="row"]') || el.closest('[data-id]') || el.parentElement;
          if (row) rows.add(row);
        });
        return Array.from(rows);
      }

      // Strategy 3: role=row
      els = mainPanel.querySelectorAll('[role="row"]');
      if (els.length > 0) return Array.from(els);

      // Strategy 4: focusable-list-item
      els = mainPanel.querySelectorAll('.focusable-list-item');
      if (els.length > 0) return Array.from(els);

      // Strategy 5: copyable-area descendants
      const copyableArea = mainPanel.querySelector('.copyable-area');
      if (copyableArea) {
        const msgList = copyableArea.querySelector('[role="application"]') || copyableArea;
        els = msgList.querySelectorAll(':scope > div > div > div[class]');
        if (els.length > 0) return Array.from(els);
      }

      // Strategy 6: text spans walked up to containers
      const textSpans = mainPanel.querySelectorAll(
        'span.selectable-text, [data-testid="msg-text"], span[dir="ltr"], span[dir="auto"]'
      );
      const containers = new Set();
      textSpans.forEach(span => {
        let parent = span.parentElement;
        for (let i = 0; i < 10 && parent && parent !== mainPanel; i++) {
          if (parent.getAttribute('data-id') ||
              parent.classList.contains('message-in') ||
              parent.classList.contains('message-out') ||
              parent.getAttribute('role') === 'row' ||
              parent.querySelector('[data-testid="msg-container"]')) {
            containers.add(parent);
            break;
          }
          parent = parent.parentElement;
        }
      });
      if (containers.size > 0) return Array.from(containers);

      // Strategy 7: Broad data-testid containing "msg" — catches renamed
      // containers like "msg-bubble", "msg-wrapper", etc.
      const msgTestIds = mainPanel.querySelectorAll('[data-testid*="msg"]');
      const msgRows = new Set();
      msgTestIds.forEach(el => {
        const testid = el.getAttribute('data-testid');
        if (testid.includes('meta') || testid.includes('reaction') ||
            testid.includes('system') || testid.includes('date')) return;
        const row = el.closest('[role="row"]') || el.closest('[data-id]') || el.parentElement?.parentElement;
        if (row && row !== mainPanel) msgRows.add(row);
      });
      if (msgRows.size > 0) {
        log(`findMessageElements: Strategy 7 matched ${msgRows.size} via [data-testid*="msg"]`);
        return Array.from(msgRows);
      }

      // Strategy 8: Direct children of conversation body with content
      const convBody = mainPanel.querySelector('[data-testid="conversation-panel-body"]')
                    || mainPanel.querySelector('[role="application"]')
                    || (scrollContainer || null);
      if (convBody) {
        const children = convBody.querySelectorAll(':scope > div > div');
        const candidates = [];
        for (const child of children) {
          const text = child.textContent?.trim();
          if (text && text.length > 2 && !isDateLikeString(text)) {
            candidates.push(child);
          }
        }
        if (candidates.length > 0) {
          log(`findMessageElements: Strategy 8 matched ${candidates.length} via structural fallback`);
          return candidates;
        }
      }

      return Array.from(containers);
    }

    // --- Determine scroll positions ---
    let positions;
    if (scrollContainer) {
      scrollContainer.scrollTop = 0;
      await sleep(1000);

      const totalHeight = scrollContainer.scrollHeight;
      const viewportHeight = scrollContainer.clientHeight;
      // Step by 50 % of viewport to guarantee overlap (no message is missed
      // because it straddled two viewports).
      const step = Math.max(Math.floor(viewportHeight * 0.5), 100);

      positions = [];
      for (let pos = 0; pos <= totalHeight; pos += step) {
        positions.push(pos);
      }
      // Always include the very bottom
      if (positions[positions.length - 1] < totalHeight) {
        positions.push(totalHeight);
      }
    } else {
      // No scroll container — single extraction pass on the static DOM.
      positions = [null];
    }

    log(`Starting scroll-and-extract: ${positions.length} positions, totalHeight=${scrollContainer?.scrollHeight}`);

    // Wait for the DOM to stabilise before extracting.  If the chat just
    // switched, message elements may not have rendered yet.
    {
      let readyRetries = 0;
      while (readyRetries < 5 && findMessageElements().length === 0) {
        log(`Waiting for message elements to appear (attempt ${readyRetries + 1}/5)`);
        await sleep(2000);
        readyRetries++;
      }
    }

    // Diagnostic: log what findMessageElements sees at the first position
    {
      const diagEls = findMessageElements();
      log(`DIAG: findMessageElements() returned ${diagEls.length} elements at start`);
      if (diagEls.length === 0) {
        // Log what IS in #main so we can see why nothing was found
        const allTestIds = Array.from(mainPanel.querySelectorAll('[data-testid]')).slice(0, 15)
          .map(e => e.getAttribute('data-testid'));
        const allRoles = Array.from(mainPanel.querySelectorAll('[role]')).slice(0, 10)
          .map(e => e.getAttribute('role'));
        log(`DIAG: #main data-testids: [${allTestIds.join(', ')}]`);
        log(`DIAG: #main roles: [${allRoles.join(', ')}]`);
        log(`DIAG: #main innerHTML length: ${mainPanel.innerHTML.length}`);
        log(`DIAG: #main textContent preview: "${mainPanel.textContent?.substring(0, 200)}"`);
      }
    }

    for (let i = 0; i < positions.length; i++) {
      if (scrollContainer && positions[i] !== null) {
        scrollContainer.scrollTop = positions[i];
        // IMP-06: Adaptive delay — resolves early when DOM settles, max 2 s.
        // Also triggers lazy-load of images so blob: URLs become available.
        await adaptiveSleep(2000, options?._signal);
      }

      // --- Date dividers visible in the current viewport ---
      const dateDividers = buildDateMap(mainPanel);
      if (dateDividers.length > 0) {
        lastKnownDateString = dateDividers[dateDividers.length - 1].dateString;
      }

      // --- Message elements in the current viewport ---
      const messageElements = findMessageElements();

      for (const msgEl of messageElements) {
        const rawId = msgEl.getAttribute('data-id') || '';

        // Deduplicate
        if (rawId && seenIds.has(rawId)) continue;
        if (rawId) seenIds.add(rawId);

        // System message handling
        if (isSystemMessage(msgEl)) {
          const text = msgEl.textContent?.trim();
          if (text) {
            systemMessages.push({
              type: classifySystemMessage(text),
              content: text
            });
          }
          continue;
        }

        // Resolve the date: prefer the divider directly preceding this
        // element; fall back to the last date we saw while scrolling.
        const dateEntry = getDateForElement(msgEl, dateDividers);
        if (dateEntry) {
          lastKnownDateString = dateEntry.dateString;
        }
        const effectiveDateEntry = dateEntry
          || (lastKnownDateString ? { dateString: lastKnownDateString } : null);

        const message = await extractMessageData(msgEl, options, effectiveDateEntry);
        if (message) {
          // IMP-03: Secondary dedup by content hash for messages with
          // missing or synthetic IDs (albums, generated containers).
          const hash = messageContentHash(message);
          if (!rawId && seenHashes.has(hash)) {
            continue; // duplicate without a stable ID
          }
          seenHashes.add(hash);

          messages.push(message);
          // If the message itself yielded a date (via strategies A–C3),
          // feed it back so subsequent dateless messages can inherit it.
          if (message.date) {
            lastKnownDateString = message.date;
          }
        }
      }

      // Progress (throttled to every 5 positions)
      if (i % 5 === 0 || i === positions.length - 1) {
        sendProgress(
          28 + ((i + 1) / positions.length * 47),
          'Extracting messages...',
          `Scanning section ${i + 1}/${positions.length} (${messages.length} messages)`
        );
      }
    }

    // --- Post-process: propagate dates to messages that are missing them ---
    // WhatsApp doesn't always render date metadata on every element.  If a
    // message has no date but its neighbours do, use the nearest neighbour's
    // date.  This is safe because messages within the same scroll position are
    // chronologically adjacent.
    let lastGoodDate = null;
    for (let m = 0; m < messages.length; m++) {
      if (messages[m].date) {
        lastGoodDate = messages[m].date;
      } else if (lastGoodDate && messages[m].timestamp) {
        messages[m].date = lastGoodDate;
        messages[m].dateTime = combineDateAndTime(lastGoodDate, messages[m].timestamp);
        log(`Post-process: filled date for msg ${m} from earlier neighbour: ${lastGoodDate}`);
      }
    }
    // Reverse pass: fill early messages that had no predecessor with a date
    let nextGoodDate = null;
    for (let m = messages.length - 1; m >= 0; m--) {
      if (messages[m].date) {
        nextGoodDate = messages[m].date;
      } else if (nextGoodDate && messages[m].timestamp) {
        messages[m].date = nextGoodDate;
        messages[m].dateTime = combineDateAndTime(nextGoodDate, messages[m].timestamp);
        log(`Post-process: filled date for msg ${m} from later neighbour: ${nextGoodDate}`);
      }
    }

    // Sort by dateTime (ISO string) — fixes cross-day ordering
    messages.sort((a, b) => {
      if (a.dateTime && b.dateTime) return a.dateTime.localeCompare(b.dateTime);
      return 0;
    });

    log(`Scroll-and-extract complete: ${messages.length} messages, ${systemMessages.length} system messages`);
    return { messages, systemMessages };
  }

  // Simulate a realistic click using PointerEvents (React 18+) followed by
  // MouseEvents and a native .click() call.  Coordinates are derived from
  // the element's bounding rect so the events land inside the target.
  function simulateClick(el) {
    // BUG-09: Find the innermost interactive element first — React's fiber
    // tree may only listen on <a>, [role="button"], or [tabindex] nodes.
    const inner = el.querySelector('a, [role="button"], [tabindex]');
    const target = inner || el;

    const rect = target.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;

    const commonOpts = {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: x,
      clientY: y,
      screenX: x + window.screenX,
      screenY: y + window.screenY,
      button: 0,
      buttons: 1,
      detail: 1,
    };

    const pointerOpts = {
      ...commonOpts,
      pointerId: 1,
      pointerType: 'mouse',
      isPrimary: true,
      width: 1,
      height: 1,
      pressure: 0.5,
    };

    // Phase 1: PointerEvents (React 18+ uses these)
    target.dispatchEvent(new PointerEvent('pointerdown', pointerOpts));
    target.dispatchEvent(new PointerEvent('pointerup', pointerOpts));

    // Phase 2: MouseEvents (fallback for older React builds)
    target.dispatchEvent(new MouseEvent('mousedown', commonOpts));
    target.dispatchEvent(new MouseEvent('mouseup', commonOpts));

    // Phase 3: Native click (triggers browser's own click pipeline)
    target.click();

    // Phase 4: If we targeted an inner element, also fire on the original
    if (inner && inner !== el) {
      el.click();
    }
  }

  // BUG-09: Keyboard-based click fallback for when pointer events don't
  // propagate through React's synthetic event system.
  function simulateKeyboardClick(el) {
    el.focus();
    const opts = { bubbles: true, cancelable: true, key: 'Enter', code: 'Enter', keyCode: 13, which: 13 };
    el.dispatchEvent(new KeyboardEvent('keydown', opts));
    el.dispatchEvent(new KeyboardEvent('keyup', opts));
  }

  // Open a specific chat by name.
  // Returns true if the chat was opened, false otherwise.
  async function openChat(chatIndex, chatName) {
    log(`openChat: looking for "${chatName}" (index=${chatIndex})`);

    // Strategy: find the title span that shows the chat name, then walk up
    // to the nearest clickable ancestor.  This is more reliable than clicking
    // outer containers because WhatsApp's React handlers are on the
    // list-item-link or the listitem/row wrapper, NOT on cell-frame-container.
    const sidePanel = document.querySelector('#pane-side');
    if (!sidePanel) {
      log('ERROR: #pane-side not found');
      return false;
    }

    // Strategy: Iterate over list items to find a matching name (title or text).
    // This matches the logic in getChatList() to ensure we can open what we found.
    let clickTarget = null;
    const containerSelectors = [
      '[data-testid="cell-frame-container"]',
      '[data-testid="list-item-link"]',
      '#pane-side [role="listitem"]',
      '#pane-side [role="row"]'
    ];

    for (const selector of containerSelectors) {
      const elements = document.querySelectorAll(selector);
      for (const el of elements) {
        // Extract name from this row
        let name = '';
        const titleEl = el.querySelector('[title]');
        if (titleEl) {
          name = titleEl.getAttribute('title');
        }

        if (!name) {
          // Fallback to text content (e.g. for contacts without a status/title set)
          const spanEl = el.querySelector('span[dir="auto"], span[title]');
          if (spanEl) {
            name = spanEl.getAttribute('title') || spanEl.textContent;
          }
        }

        if (name && name.trim() === chatName) {
          clickTarget = el;
          log(`openChat: found "${chatName}" via selector "${selector}"`);
          break;
        }
      }
      if (clickTarget) break;
    }

    // BUG-05: Removed index-based fallback. Incoming messages can reorder
    // the sidebar, making index unreliable. Name-based matching above is
    // the only safe approach.
    if (!clickTarget) {
      log(`WARNING: openChat could not find "${chatName}" by name in any selector strategy`);
    }

    if (!clickTarget) {
      log(`ERROR: Could not find chat "${chatName}" in sidebar`);
      return false;
    }

    // Capture the CURRENT header before clicking so we can detect a change.
    const preClickHeader = (() => {
      const header = document.querySelector('#main header');
      return header?.querySelector('[title]')?.getAttribute('title') ||
             header?.querySelector('span[dir="auto"]')?.textContent?.trim() || '';
    })();

    // Capture current message IDs to verify DOM actually changes
    const preClickMessageIds = new Set(
      Array.from(document.querySelectorAll('#main [data-id]'))
        .slice(0, 5)
        .map(el => el.getAttribute('data-id'))
    );

    // Scroll the chat into view so click coordinates are valid
    clickTarget.scrollIntoView({ block: 'center', behavior: 'instant' });
    await sleep(300);

    // Click using simulated events to trigger React handlers
    simulateClick(clickTarget);

    // Wait for header change using MutationObserver
    let headerChanged = await waitForHeaderChange(chatName, preClickHeader, 10000);

    if (!headerChanged) {
      // Escalation: try clicking the title span directly
      log('openChat: first click failed, retrying on title span');
      for (const s of sidePanel.querySelectorAll('span[title]')) {
        if (s.getAttribute('title') === chatName) {
          s.scrollIntoView({ block: 'center', behavior: 'instant' });
          await sleep(200);
          simulateClick(s);
          break;
        }
      }
      headerChanged = await waitForHeaderChange(chatName, preClickHeader, 8000);
    }

    if (!headerChanged) {
      // Escalation 3: try clicking the parent of the original target
      log('openChat: second attempt failed, retrying on parent element');
      const parent = clickTarget.parentElement;
      if (parent && parent !== sidePanel) {
        simulateClick(parent);
        headerChanged = await waitForHeaderChange(chatName, preClickHeader, 6000);
      }
    }

    if (!headerChanged) {
      // BUG-09 escalation 4: keyboard-based fallback
      log('openChat: trying keyboard click fallback');
      simulateKeyboardClick(clickTarget);
      headerChanged = await waitForHeaderChange(chatName, preClickHeader, 4000);
    }

    // Final fallback: verify via message content change instead of header.
    // This handles cases where WhatsApp removed the <header> element or
    // changed the title/span structure so header verification always fails.
    if (!headerChanged) {
      log('openChat: header verification failed, trying message-based fallback');
      // Give the DOM a moment to settle after the last click attempt
      await sleep(1500);
      const postClickIds = new Set(
        Array.from(document.querySelectorAll('#main [data-id]'))
          .slice(0, 5)
          .map(el => el.getAttribute('data-id'))
      );
      const hasNewMessages = [...postClickIds].some(id => id && !preClickMessageIds.has(id));
      if (hasNewMessages || (preClickMessageIds.size === 0 && postClickIds.size > 0)) {
        log('openChat: message content changed — considering chat switch successful');
        headerChanged = true;
      }
    }

    // Final-final fallback: check if #main textContent contains the chat name
    // (covers the case where the chat was already open before clicking).
    if (!headerChanged) {
      const mainText = document.querySelector('#main')?.textContent?.substring(0, 1000) || '';
      if (mainText.includes(chatName)) {
        log('openChat: chat name found in #main text — considering already open');
        headerChanged = true;
      }
    }

    if (!headerChanged) {
      log(`ERROR: Chat did not switch to "${chatName}" after all attempts`);
      return false;
    }

    // After header changes, verify the conversation body has repopulated
    // with new message IDs (not stale content from the previous chat).
    const bodyReady = await waitForNewMessages(preClickMessageIds, 8000);
    if (!bodyReady) {
      log('WARNING: Conversation body did not repopulate — proceeding anyway');
    }

    log(`openChat: confirmed switch to "${chatName}"`);
    return true;
  }

  // Wait for the conversation header to show a specific chat name.
  // Uses MutationObserver to detect actual DOM changes rather than polling.
  function waitForHeaderChange(expectedName, previousName, timeoutMs) {
    return new Promise((resolve) => {
      // Try <header> first, then fall back to the first child section of #main
      // (WhatsApp may have replaced <header> with a plain <div>).
      const mainEl = document.querySelector('#main');
      const headerContainer = mainEl?.querySelector('header')
                           || mainEl?.firstElementChild;
      if (!headerContainer) {
        resolve(false);
        return;
      }

      const getTitle = () => {
        // Approach 1: [title] attribute (old WhatsApp)
        const titleEl = headerContainer.querySelector('[title]');
        if (titleEl) return titleEl.getAttribute('title');

        // Approach 2: First meaningful span text (skip timestamps/status)
        const spans = headerContainer.querySelectorAll('span[dir="auto"], span[dir="ltr"]');
        for (const span of spans) {
          const text = span.textContent?.trim();
          if (text && text.length > 1 && text.length < 100
              && !/^\d{1,2}:\d{2}/.test(text)
              && !/^(click here|tap here|online|offline|last seen|typing)/i.test(text)) {
            return text;
          }
        }

        // Approach 3: img alt (contact photo alt text)
        const img = headerContainer.querySelector('img[alt]');
        if (img?.alt) return img.alt;

        return '';
      };

      // Use includes() matching: the header may contain extra text like
      // status/last-seen alongside the name.
      const matches = (title) => {
        if (!title) return false;
        return title === expectedName
            || title.includes(expectedName)
            || expectedName.includes(title);
      };

      // Check immediately — the click may have already taken effect
      if (matches(getTitle())) {
        resolve(true);
        return;
      }

      let resolved = false;
      const observer = new MutationObserver(() => {
        if (resolved) return;
        if (matches(getTitle())) {
          resolved = true;
          observer.disconnect();
          resolve(true);
        }
      });

      observer.observe(headerContainer, {
        childList: true,
        subtree: true,
        characterData: true,
        attributes: true,
      });

      setTimeout(() => {
        if (!resolved) {
          observer.disconnect();
          // Final snapshot check
          resolve(matches(getTitle()));
        }
      }, timeoutMs);
    });
  }

  // Wait until the conversation body shows message elements that differ
  // from the previous chat.  This prevents extracting stale DOM content
  // when the header has updated but messages are still loading.
  async function waitForNewMessages(oldMessageIds, timeoutMs) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const currentIds = Array.from(document.querySelectorAll('#main [data-id]'))
        .slice(0, 5)
        .map(el => el.getAttribute('data-id'));

      // At least one new ID that was NOT in the previous chat
      const hasNew = currentIds.some(id => id && !oldMessageIds.has(id));
      if (hasNew && currentIds.length > 0) {
        await sleep(500); // let the rest of the DOM settle
        return true;
      }

      // If oldMessageIds was empty (no previous chat), any IDs count
      if (oldMessageIds.size === 0 && currentIds.length > 0) {
        await sleep(500);
        return true;
      }

      await sleep(500);
    }
    return false;
  }

  // -----------------------------------------------------------------------
  // Build a date map from date-divider elements so each message gets a date
  // -----------------------------------------------------------------------
  function buildDateMap(mainPanel) {
    const dateMap = []; // Array of { element, dateString }

    // Strategy 1: Exact data-testid selectors — try ALL selectors and
    // accumulate results (don't short-circuit after the first match).
    const dividerSelectors = [
      '[data-testid="date-divider"]',
      '[data-testid="msg-date-divider"]',
    ];

    const seenEls = new Set();
    for (const selector of dividerSelectors) {
      const dividers = mainPanel.querySelectorAll(selector);
      dividers.forEach(div => {
        if (seenEls.has(div)) return;
        seenEls.add(div);
        const text = div.textContent?.trim();
        if (text) {
          dateMap.push({ element: div, dateString: text });
        }
      });
    }
    if (dateMap.length > 0) {
      log(`Date dividers found via exact selectors: ${dateMap.length}`);
      return dateMap;
    }

    // Strategy 2: Any element whose data-testid contains "date" (catches variations)
    const wildcardDividers = mainPanel.querySelectorAll('[data-testid*="date"]');
    wildcardDividers.forEach(div => {
      const text = div.textContent?.trim();
      if (text && isDateLikeString(text)) {
        dateMap.push({ element: div, dateString: text });
      }
    });
    if (dateMap.length > 0) {
      log(`Date dividers found via wildcard [data-testid*="date"]`);
      return dateMap;
    }

    // Strategy 3: Rows/divs that have NO message content (no data-id, no msg-container)
    // but do contain date-like text. Check [role="row"] and direct children of the
    // scroll container / conversation panel.
    const candidateContainers = [
      ...mainPanel.querySelectorAll('[role="row"]'),
      ...mainPanel.querySelectorAll('[role="listitem"]'),
    ];

    // Also check direct children of the conversation panel body
    const convBody = mainPanel.querySelector('[data-testid="conversation-panel-body"]')
                  || mainPanel.querySelector('[role="application"]');
    if (convBody) {
      for (const child of convBody.children) {
        candidateContainers.push(child);
      }
    }

    const seen = new Set();
    for (const el of candidateContainers) {
      if (seen.has(el)) continue;
      seen.add(el);
      // Skip actual message rows
      if (el.querySelector('[data-testid="msg-container"]')) continue;
      if (el.getAttribute('data-id') && /^(true|false)_/.test(el.getAttribute('data-id'))) continue;

      const text = el.textContent?.trim();
      if (text && isDateLikeString(text)) {
        dateMap.push({ element: el, dateString: text });
      }
    }

    if (dateMap.length > 0) {
      log(`Date dividers found via row/listitem scan: ${dateMap.length}`);
    } else {
      // Diagnostic: log the first few non-message elements to help debug
      log('WARNING: No date dividers found. Dumping non-message elements for diagnosis:');
      let count = 0;
      for (const el of candidateContainers) {
        if (!el.querySelector('[data-testid="msg-container"]') &&
            !(el.getAttribute('data-id') && /^(true|false)_/.test(el.getAttribute('data-id')))) {
          log(`  Non-msg element: tag=${el.tagName}, testid=${el.getAttribute('data-testid')}, role=${el.getAttribute('role')}, text="${el.textContent?.trim().substring(0, 80)}"`);
          count++;
          if (count >= 10) break;
        }
      }
    }

    return dateMap;
  }

  // Heuristic: is this text a date string ("1/23/2026", "YESTERDAY", "TODAY", "12/31/2025")?
  function isDateLikeString(text) {
    const lower = text.toLowerCase().trim();
    if (['today', 'yesterday', 'hoy', 'ayer'].includes(lower)) return true;
    // Match common date patterns: M/D/YYYY, D/M/YYYY, MM-DD-YYYY, YYYY-MM-DD, etc.
    if (/^\d{1,4}[\/-]\d{1,2}[\/-]\d{1,4}$/.test(lower)) return true;
    // Match "January 23, 2026" or "23 January 2026" style
    if (/^[a-z]+ \d{1,2},?\s*\d{4}$/i.test(lower)) return true;
    if (/^\d{1,2} [a-z]+ \d{4}$/i.test(lower)) return true;
    return false;
  }

  // Parse a date-divider string into a Date object
  function parseDateDivider(dateString) {
    const lower = dateString.toLowerCase().trim();
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    if (lower === 'today' || lower === 'hoy') return today;
    if (lower === 'yesterday' || lower === 'ayer') {
      today.setDate(today.getDate() - 1);
      return today;
    }

    // Handle YYYY-MM-DD explicitly to avoid UTC parsing.
    // new Date("2026-02-10") creates UTC midnight which is the PREVIOUS day
    // in west-of-UTC timezones.  Using the (year, month, day) constructor
    // creates a local-time date instead.
    const isoMatch = dateString.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (isoMatch) {
      return new Date(Number(isoMatch[1]), Number(isoMatch[2]) - 1, Number(isoMatch[3]));
    }

    // Try numeric date formats explicitly before native Date.parse,
    // because native parsing of ambiguous formats is unreliable.
    // BUG-07: Use detectedDateOrder (from HTML lang / UI strings) to
    // resolve ambiguous slash-separated dates like "5/3/2026".
    const sepMatch = dateString.match(/[\/.:-]/);
    const parts = dateString.split(/[\/.:-]/);
    if (parts.length === 3) {
      const [a, b, c] = parts.map(Number);
      if (!isNaN(a) && !isNaN(b) && !isNaN(c)) {
        const sep = sepMatch ? sepMatch[0] : '/';
        // YYYY/MM/DD or YYYY-MM-DD (already handled above for hyphens)
        if (a >= 100 && b <= 12) {
          return new Date(a, b - 1, c);
        }
        // Dot separator → D.M.YYYY (European)
        if (sep === '.' && b <= 12 && c >= 100) {
          return new Date(c, b - 1, a);
        }
        // Slash separator — use locale hint when available
        if (c >= 100) {
          if (detectedDateOrder === 'dmy' && b <= 12) {
            return new Date(c, b - 1, a); // D/M/YYYY
          }
          if (detectedDateOrder === 'mdy' && a <= 12) {
            return new Date(c, a - 1, b); // M/D/YYYY
          }
          // No locale hint — disambiguate by value range
          if (a > 12 && b <= 12) return new Date(c, b - 1, a); // D/M/YYYY
          if (b > 12 && a <= 12) return new Date(c, a - 1, b); // M/D/YYYY
          // Both ≤ 12: prefer interpretation that gives date ≤ today
          if (a <= 12 && b <= 12) {
            const today = new Date();
            const mdyDate = new Date(c, a - 1, b);
            const dmyDate = new Date(c, b - 1, a);
            if (mdyDate <= today && dmyDate > today) return mdyDate;
            if (dmyDate <= today && mdyDate > today) return dmyDate;
            return mdyDate; // default M/D/YYYY
          }
          if (a <= 12) return new Date(c, a - 1, b);
          if (b <= 12) return new Date(c, b - 1, a);
        }
      }
    }

    // Fallback: native Date.parse (for "January 23, 2026" style strings).
    // Avoid for date-only numeric strings as they may be parsed as UTC.
    const parsed = new Date(dateString);
    if (!isNaN(parsed.getTime())) return parsed;

    return null;
  }

  // Given a message element and the date dividers, find which date it falls under
  function getDateForElement(msgEl, dateMap) {
    // Walk backwards through date dividers; the message belongs to the last divider
    // that appears before it in DOM order.
    let currentDate = null;
    for (const entry of dateMap) {
      // compareDocumentPosition: DOCUMENT_POSITION_FOLLOWING (4) means entry is before msgEl
      const pos = entry.element.compareDocumentPosition(msgEl);
      if (pos & Node.DOCUMENT_POSITION_FOLLOWING) {
        currentDate = entry;
      } else {
        break;
      }
    }
    return currentDate;
  }

  // Combine a date string from a divider and a time string into ISO 8601
  function combineDateAndTime(dateString, timeString) {
    if (!dateString) return null;

    const dateObj = parseDateDivider(dateString);
    if (!dateObj) return null;

    // Parse time like "1:59 AM" or "13:59"
    const timeMatch = timeString?.match(/^(\d{1,2}):(\d{2})\s*(AM|PM)?$/i);
    if (timeMatch) {
      let hours = parseInt(timeMatch[1], 10);
      const minutes = parseInt(timeMatch[2], 10);
      const ampm = timeMatch[3];
      if (ampm) {
        if (ampm.toUpperCase() === 'PM' && hours !== 12) hours += 12;
        if (ampm.toUpperCase() === 'AM' && hours === 12) hours = 0;
      }
      dateObj.setHours(hours, minutes, 0, 0);
    }

    // Return ISO 8601 local datetime (no timezone offset to keep it simple)
    const pad = (n) => String(n).padStart(2, '0');
    return `${dateObj.getFullYear()}-${pad(dateObj.getMonth() + 1)}-${pad(dateObj.getDate())}T${pad(dateObj.getHours())}:${pad(dateObj.getMinutes())}:00`;
  }

  // Extract just the date portion as YYYY-MM-DD
  function formatDateOnly(dateString) {
    const dateObj = parseDateDivider(dateString);
    if (!dateObj) return null;
    const pad = (n) => String(n).padStart(2, '0');
    return `${dateObj.getFullYear()}-${pad(dateObj.getMonth() + 1)}-${pad(dateObj.getDate())}`;
  }

  // -----------------------------------------------------------------------
  // Detect system messages
  // -----------------------------------------------------------------------
  function isSystemMessage(el) {
    // data-testid based detection
    if (el.querySelector('[data-testid="msg-system"]')) return true;

    // data-id pattern: real messages start with "true_" or "false_"
    const dataId = el.getAttribute('data-id');
    if (dataId && !/^(true|false)_/.test(dataId)) return true;

    // No data-id: could be a system message OR an album/gallery container.
    // Album containers hold real images so we must not discard them.
    if (!dataId) {
      // Check 0: data-pre-plain-text or message-in/message-out = real message
      if (el.querySelector('[data-pre-plain-text]')) return false;
      if (el.querySelector('.message-in, .message-out')) return false;

      // Check 1: Loaded media (images with blob/media src, video, audio, docs)
      const hasMedia = el.querySelector(
        'img[src*="blob:"], img[src*="media"], video, ' +
        '[data-testid*="video"], [data-testid*="ptt"], ' +
        '[data-testid*="audio"], [data-testid*="document"]'
      );
      if (hasMedia) return false;

      // Check 2: Album / gallery testids
      if (el.querySelector('[data-testid*="album"], [data-testid*="gallery"]')) return false;

      // Check 3: Forwarded message indicator — forwarded messages are real
      if (el.querySelector('[data-testid*="forwarded"], [data-testid="forward-refreshed"]')) return false;
      const textLower = (el.textContent || '').toLowerCase();
      if (textLower.includes('forwarded')) {
        // Forwarded labels paired with timestamps are real messages
        if (/\d{1,2}:\d{2}/.test(el.textContent || '')) return false;
      }

      // Check 4: Lazy-loaded images / canvas placeholders — multiple = album
      const lazyImages = el.querySelectorAll('img, canvas');
      if (lazyImages.length >= 2) return false;

      // Check 5: Any img inside a message-like structure
      const anyImg = el.querySelector('img');
      if (anyImg) {
        const imgParent = anyImg.closest('[class*="message"], [data-testid*="msg"]');
        if (imgParent) return false;
      }

      // Check 6: Has msg-container somewhere in subtree
      if (el.querySelector('[data-testid="msg-container"]')) return false;

      // Check 7: Substantial content heuristic — sender + timestamp pattern
      const hasSubstantialContent = el.querySelectorAll('span[dir="auto"]').length >= 3;
      if (hasSubstantialContent && /\d{1,2}:\d{2}/.test(el.textContent || '')) return false;

      // None of the above matched — treat as system message
      return true;
    }

    // Content-based detection: some system messages (encryption notices,
    // member additions) carry a regular-looking data-id but are not real
    // user messages.
    const text = (el.textContent || '').trim().toLowerCase();
    if (text.includes('messages and calls are end-to-end encrypted')) return true;

    // Group info header panel — appears when scrolled to the very top.
    // Contains "Group info" text and member counts.
    if (text.includes('group info') && (text.includes('members') || text.includes('created'))) return true;

    return false;
  }

  function classifySystemMessage(text) {
    const lower = (text || '').toLowerCase();
    if (lower.includes('end-to-end encrypted') || lower.includes('messages and calls are')) {
      return 'encryption_notice';
    }
    if (lower.includes('created') || lower.includes('added') || lower.includes('removed') || lower.includes('left')) {
      return 'group_info';
    }
    if (lower.includes('changed') || lower.includes('turned on') || lower.includes('turned off')) {
      return 'setting_change';
    }
    return 'system';
  }

  // NOTE: extractMessages has been replaced by scrollAndExtractMessages above,
  // which combines scrolling and extraction in a single pass to handle
  // WhatsApp's virtualised message list.

  // -----------------------------------------------------------------------
  // Extract data from a single message element
  // -----------------------------------------------------------------------
  async function extractMessageData(msgEl, options, dateEntry) {
    const message = {
      id: msgEl.getAttribute('data-id') || generateId(),
      type: 'text',
      content: '',
      timestamp: '',
      dateTime: null,
      date: null,
      sender: '',
      isOutgoing: false,
      isForwarded: false,
      isDeleted: false,
      media: null,
      quotedMessage: null,
      reactions: null
    };

    // Check element and parent classes for direction
    const checkOutgoing = (el) => {
      if (!el) return false;
      const classes = el.className || '';
      return classes.includes('message-out');
    };

    message.isOutgoing = checkOutgoing(msgEl) ||
                          checkOutgoing(msgEl.parentElement) ||
                          checkOutgoing(msgEl.parentElement?.parentElement) ||
                          msgEl.querySelector('[data-icon="msg-dblcheck"], [data-icon="msg-check"]') !== null ||
                          msgEl.querySelector('[data-testid="msg-dblcheck"], [data-testid="msg-check"]') !== null;

    // Detect forwarded messages
    if (msgEl.querySelector('[data-testid*="forwarded"], [data-testid="forward-refreshed"]')) {
      message.isForwarded = true;
    }

    // Check for deleted
    if (msgEl.querySelector('[data-icon="recalled"]')) {
      message.isDeleted = true;
      message.content = '[Message was deleted]';
      message.type = 'deleted';
      if (!options.includeDeleted) return null;
    }

    // --- TASK 1: Improved sender extraction ---
    let senderEl = null;

    // Priority 1: data-testid="msg-author-title" (most stable in group chats)
    senderEl = msgEl.querySelector('[data-testid="msg-author-title"]');
    if (senderEl) log(`Sender found via Priority 1 (msg-author-title): "${senderEl.textContent?.trim()}"`);

    // Priority 2: Original selector
    if (!senderEl) {
      senderEl = msgEl.querySelector('[data-testid="author"]');
      if (senderEl) log(`Sender found via Priority 2 (author): "${senderEl.textContent?.trim()}"`);
    }

    // Priority 3: aria-label with @ (contact card style)
    if (!senderEl) {
      senderEl = msgEl.querySelector('span[aria-label*="@"]');
      if (senderEl) log(`Sender found via Priority 3 (aria-label @): "${senderEl.textContent?.trim()}"`);
    }

    // Priority 4: Clickable contact-name wrapper before the message body
    if (!senderEl) {
      const contactWrapper = msgEl.querySelector('[data-testid="msg-author"], [data-testid="msg-author-name"]');
      if (contactWrapper) {
        senderEl = contactWrapper;
        log(`Sender found via Priority 4 (msg-author): "${senderEl.textContent?.trim()}"`);
      }
    }

    // Priority 5: First short <span dir="auto"> INSIDE msg-container that is NOT
    // inside the message body, meta, or quoted-message sections.
    // In group chats the sender is a colored span at the TOP of the bubble.
    if (!senderEl) {
      const msgContainer = msgEl.querySelector('[data-testid="msg-container"]') || msgEl;
      // Anchor: the first content element (text, image, video, audio, doc)
      const contentAnchor = msgContainer.querySelector(
        '[data-testid="msg-text"], img, video, [data-testid*="ptt"], [data-testid*="audio"], [data-testid*="document"], [data-testid*="video"]'
      );
      const allSpans = msgContainer.querySelectorAll('span[dir="auto"]');
      for (const span of allSpans) {
        const text = span.textContent?.trim();
        // Skip empty, time-like, and overly long text (message body)
        if (!text || text.length === 0 || text.length > 60) continue;
        if (/^\d{1,2}:\d{2}(\s*[AP]M)?$/i.test(text)) continue;
        // Must NOT be inside content/meta areas
        if (span.closest('[data-testid="msg-text"]')) continue;
        if (span.closest('[data-testid="msg-meta"]')) continue;
        if (span.closest('[data-testid="quoted-message"]')) continue;
        // If we have a content anchor, sender must appear before it in DOM
        if (contentAnchor) {
          const pos = contentAnchor.compareDocumentPosition(span);
          if (!(pos & Node.DOCUMENT_POSITION_PRECEDING)) continue;
        }
        senderEl = span;
        log(`Sender found via Priority 5 (span inside bubble): "${text}"`);
        break;
      }
    }

    if (senderEl) {
      message.sender = senderEl.textContent?.trim() || senderEl.getAttribute('aria-label')?.replace(/:$/, '') || '';
      // Mark so text extraction skips it
      senderEl.setAttribute('data-wa-sender-extracted', 'true');
    }

    // Priority 6: Extract sender from data-pre-plain-text attribute.
    // Format: "[time, date] Sender Name: " — the sender is between ] and the trailing :
    if (!message.sender) {
      const prePlainEl = msgEl.querySelector('[data-pre-plain-text]');
      if (prePlainEl) {
        const attr = prePlainEl.getAttribute('data-pre-plain-text');
        const senderMatch = attr.match(/\]\s*([^:]+):\s*$/);
        if (senderMatch) {
          const extractedSender = senderMatch[1].trim();
          if (extractedSender && extractedSender.length > 0 && extractedSender.length <= 80) {
            message.sender = extractedSender;
            log(`Sender found via Priority 6 (data-pre-plain-text): "${extractedSender}"`);
          }
        }
      }
    }

    // Priority 7: Extract sender from aria-label on the message row.
    // WhatsApp sometimes includes the sender in the row's aria-label.
    if (!message.sender && !message.isOutgoing) {
      const rowEl = msgEl.closest('[role="row"]') || msgEl;
      const ariaLabel = rowEl.getAttribute('aria-label') || '';
      // Typical format: "Sender Name, 12:48 PM" or just the sender name
      if (ariaLabel) {
        // Strip trailing timestamp patterns like ", 12:48 PM" or ", 14:48"
        const cleaned = ariaLabel.replace(/,\s*\d{1,2}:\d{2}(\s*[AP]M)?\s*$/i, '').trim();
        if (cleaned && cleaned.length > 0 && cleaned.length <= 80) {
          message.sender = cleaned;
          log(`Sender found via Priority 7 (row aria-label): "${cleaned}"`);
        }
      }
    }

    if (!message.sender && !message.isOutgoing) {
      // Diagnostic: dump the first few spans/elements so we can see the DOM structure
      const msgContainer = msgEl.querySelector('[data-testid="msg-container"]') || msgEl;
      const diagSpans = msgContainer.querySelectorAll('span[dir="auto"], span[dir="ltr"]');
      if (diagSpans.length > 0) {
        const diagInfo = Array.from(diagSpans).slice(0, 5).map(s =>
          `"${s.textContent?.trim().substring(0, 40)}" (parent-testid=${s.parentElement?.getAttribute('data-testid') || 'none'}, closest-testid=${s.closest('[data-testid]')?.getAttribute('data-testid') || 'none'})`
        );
        log(`Sender extraction FAILED for msg ${message.id?.substring(0, 30)}. First spans:`, diagInfo);
      }
    }

    if (!message.sender) {
      message.sender = message.isOutgoing ? 'You' : '';
    }

    // --- Get text content (skip sender element) ---
    // Detect media early: for media messages we only try the most targeted
    // text strategy (msg-text = caption) to avoid scraping UI decoration
    // or CSS injected by other browser extensions (e.g. Adobe Express).
    const _hasMediaEl = !!(
      msgEl.querySelector('img[src*="blob:"], img[src*="media"]') ||
      msgEl.querySelector('[data-testid*="video"], video') ||
      msgEl.querySelector('[data-testid*="ptt"], [data-testid*="audio"]') ||
      msgEl.querySelector('[data-testid*="document"]')
    );

    if (!message.isDeleted) {
      let textEl = null;

      // Strategy 1: data-testid (most stable — works for text and captions)
      // WhatsApp renamed msg-text → selectable-text circa 2026.
      textEl = msgEl.querySelector('[data-testid="msg-text"]')
            || msgEl.querySelector('[data-testid="selectable-text"]');

      // Strategies 2–5 only for non-media messages to prevent garbage.
      if (!textEl && !_hasMediaEl) {
        // Strategy 2: Known class combinations
        textEl = msgEl.querySelector('span.selectable-text.copyable-text');
        if (!textEl) {
          textEl = msgEl.querySelector('span.selectable-text');
        }

        // Strategy 3: copyable-text class
        if (!textEl) {
          textEl = msgEl.querySelector('.copyable-text [dir]');
        }

        // Strategy 4: span[dir] with filtering
        if (!textEl) {
          const spans = msgEl.querySelectorAll('span[dir="ltr"], span[dir="auto"]');
          for (const span of spans) {
            const text = span.textContent?.trim();
            if (text && text.length > 1 && text.length < 5000
                && !/^\d{1,2}:\d{2}(\s*[AP]M)?$/i.test(text)
                && !/\{\s*[\w-]+\s*:/.test(text) // skip CSS injected by extensions
                && !span.closest('[data-testid="quoted-message"]')
                && !span.closest('[data-testid="msg-meta"]')
                && !span.closest('[data-testid="author"]')
                && !span.hasAttribute('data-wa-sender-extracted')
                && !span.closest('[data-wa-sender-extracted]')) {
              textEl = span;
              break;
            }
          }
        }

        // Strategy 5: Last resort — innerText from msg container
        if (!textEl) {
          const bubble = msgEl.querySelector('[data-testid="msg-container"]') || msgEl;
          const clone = bubble.cloneNode(true);
          // Strip WhatsApp UI meta AND injected style/script from extensions
          clone.querySelectorAll(
            '[data-testid="msg-meta"], [data-testid="quoted-message"], ' +
            '[data-testid="reactions"], [data-testid="author"], ' +
            '[data-testid="msg-author-title"], [data-wa-sender-extracted], ' +
            '[data-testid*="forwarded"], [data-testid="forward-refreshed"], ' +
            '[data-testid*="tail"], ' +
            'style, script, link[rel="stylesheet"]'
          ).forEach(el => el.remove());
          const text = clone.innerText?.trim();
          if (text && text.length > 0) {
            message.content = text;
          }
        }
      }

      if (textEl && !message.content) {
        message.content = textEl.textContent?.trim() || '';
      }
    }

    // --- Timestamp + date extraction (multi-strategy) ---
    let extractedTime = '';
    let extractedDateStr = null; // raw date string, e.g. "2/21/2026"

    // Strategy A: data-pre-plain-text attribute — "[time, date] sender: "
    // This is the most reliable source when present.
    const prePlainEl = msgEl.querySelector('[data-pre-plain-text]');
    if (prePlainEl) {
      const attr = prePlainEl.getAttribute('data-pre-plain-text');
      const pptMatch = attr.match(/\[([^\],]+),\s*([^\]]+)\]/);
      if (pptMatch) {
        extractedTime = pptMatch[1].trim();
        extractedDateStr = pptMatch[2].trim();
        log(`Date via data-pre-plain-text: time="${extractedTime}", date="${extractedDateStr}"`);
      }
    }

    // Strategy B: title attribute on the time element inside msg-meta
    // WhatsApp often stores the full "date, time" in the title tooltip.
    if (!extractedDateStr) {
      const metaEl = msgEl.querySelector('[data-testid="msg-meta"]');
      if (metaEl) {
        const candidates = [metaEl, ...metaEl.querySelectorAll('span, div')];
        for (const el of candidates) {
          const title = el.getAttribute('title') || '';
          // Match "2/21/2026, 2:48:32 PM", "21.02.2026, 14:48", etc.
          const titleMatch = title.match(
            /(\d{1,4}[\/.:-]\d{1,2}[\/.:-]\d{1,4})[,\s]+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)/i
          );
          if (titleMatch) {
            extractedDateStr = titleMatch[1];
            if (!extractedTime) {
              // Strip seconds if present: "2:48:32 PM" → "2:48 PM"
              extractedTime = titleMatch[2]
                .replace(/:\d{2}(\s*(?:AM|PM))/i, '$1')
                .replace(/:\d{2}$/, '');
            }
            log(`Date via title attr: date="${extractedDateStr}", time="${extractedTime}"`);
            break;
          }
        }
      }
    }

    // Strategy C: aria-label on the message row (may contain full datetime)
    if (!extractedDateStr) {
      const rowEl = msgEl.closest('[role="row"]') || msgEl;
      const ariaLabel = rowEl.getAttribute('aria-label') || '';
      const ariaMatch = ariaLabel.match(/(\d{1,4}[\/.:-]\d{1,2}[\/.:-]\d{1,4})/);
      if (ariaMatch) {
        extractedDateStr = ariaMatch[1];
        log(`Date via aria-label: date="${extractedDateStr}"`);
      }
    }

    // Strategy C2: aria-label on spans inside msg-meta (WhatsApp sometimes
    // stores full date+time in the aria-label of the time indicator span).
    if (!extractedDateStr) {
      const metaEl = msgEl.querySelector('[data-testid="msg-meta"]');
      if (metaEl) {
        const ariaSpans = metaEl.querySelectorAll('span[aria-label]');
        for (const span of ariaSpans) {
          const al = span.getAttribute('aria-label') || '';
          const alMatch = al.match(/(\d{1,4}[\/.:-]\d{1,2}[\/.:-]\d{1,4})[,\s]+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)/i);
          if (alMatch) {
            extractedDateStr = alMatch[1];
            if (!extractedTime) extractedTime = alMatch[2].replace(/:\d{2}(\s*(?:AM|PM))/i, '$1').replace(/:\d{2}$/, '');
            log(`Date via meta span aria-label: date="${extractedDateStr}"`);
            break;
          }
        }
      }
    }

    // Strategy C3: <time> element with datetime attribute
    if (!extractedDateStr) {
      const timeEl = msgEl.querySelector('time[datetime]');
      if (timeEl) {
        const dt = timeEl.getAttribute('datetime');
        if (dt) {
          const dtMatch = dt.match(/(\d{4}-\d{2}-\d{2})/);
          if (dtMatch) {
            extractedDateStr = dtMatch[1];
            log(`Date via <time datetime>: date="${extractedDateStr}"`);
          }
        }
      }
    }

    // Strategy D: bare time from any span (original fallback)
    if (!extractedTime) {
      const timeSpans = msgEl.querySelectorAll('span');
      for (const span of timeSpans) {
        const text = span.textContent?.trim();
        if (text && /^\d{1,2}:\d{2}(\s*[AP]M)?$/i.test(text)) {
          extractedTime = text;
          break;
        }
      }
    }

    if (options.includeTimestamps) {
      message.timestamp = extractedTime;
    }

    // Build full date and dateTime fields
    if (extractedDateStr) {
      // Date found directly on the message element
      const dateObj = parseDateDivider(extractedDateStr);
      if (dateObj) {
        const pad = (n) => String(n).padStart(2, '0');
        message.date = `${dateObj.getFullYear()}-${pad(dateObj.getMonth() + 1)}-${pad(dateObj.getDate())}`;
        message.dateTime = combineDateAndTime(extractedDateStr, extractedTime);
      }
    } else if (dateEntry && dateEntry.dateString) {
      // Fall back to date divider
      message.date = formatDateOnly(dateEntry.dateString);
      message.dateTime = combineDateAndTime(dateEntry.dateString, extractedTime);
    } else if (extractedTime) {
      // No date at all — record raw time
      message.timestamp = extractedTime;
    }

    // Diagnostic: log ALL dateless messages so we can identify which DOM
    // attributes are available for date extraction.
    if (!message.date) {
      const meta = msgEl.querySelector('[data-testid="msg-meta"]');
      const metaSpans = meta ? Array.from(meta.querySelectorAll('span')) : [];
      const row = msgEl.closest('[role="row"]');
      const timeEl = msgEl.querySelector('time[datetime]');
      log(`DATE DIAGNOSTIC for ${message.id?.substring(0, 40)}:`,
        `\n  pre-plain-text=${prePlainEl?.getAttribute('data-pre-plain-text')?.substring(0, 80) || 'NONE'}`,
        `\n  meta-el=${meta ? 'YES' : 'NO'}`,
        `\n  meta-spans=[${metaSpans.slice(0, 5).map(s =>
          `text="${s.textContent?.trim()}" title="${s.getAttribute('title') || ''}" aria="${s.getAttribute('aria-label') || ''}"`
        ).join(', ')}]`,
        `\n  row-aria="${row?.getAttribute('aria-label')?.substring(0, 100) || 'NONE'}"`,
        `\n  time-el=${timeEl ? 'datetime=' + timeEl.getAttribute('datetime') : 'NONE'}`,
        `\n  msg-testids=[${Array.from(msgEl.querySelectorAll('[data-testid]')).slice(0, 8).map(e => e.getAttribute('data-testid')).join(', ')}]`
      );
    }

    // Check for media
    const imageEl = msgEl.querySelector('img[src*="blob:"], img[src*="media"]');
    const videoEl = msgEl.querySelector('[data-testid*="video"], video');
    const audioEl = msgEl.querySelector('[data-testid*="ptt"], [data-testid*="audio"]');
    const docEl = msgEl.querySelector('[data-testid*="document"]');

    if (imageEl && imageEl.src) {
      message.type = 'image';
      message.media = { type: 'image', src: imageEl.src };
      if (!message.content) message.content = '[Image]';
      // BUG-04: Fetch blob URL immediately while the element is visible in
      // the viewport.  WhatsApp aggressively revokes blob URLs for off-screen
      // images, so deferring to extractMedia() often fails.
      if (imageEl.src.startsWith('blob:')) {
        try {
          const resp = await fetch(imageEl.src);
          const blob = await resp.blob();
          message.media._inlineBase64 = await blobToBase64(blob);
        } catch (_e) {
          log('BUG-04: Inline blob fetch failed (may already be revoked):', _e.message);
        }
      }
    } else if (videoEl) {
      message.type = 'video';
      message.media = { type: 'video' };
      if (!message.content) message.content = '[Video]';
    } else if (audioEl) {
      message.type = 'audio';
      message.media = { type: 'audio' };
      if (!message.content) message.content = '[Voice message]';
    } else if (docEl) {
      message.type = 'document';
      message.media = { type: 'document' };
      if (!message.content) message.content = '[Document]';
    }

    // Get reactions
    const reactionEl = msgEl.querySelector('[data-testid="reactions"]');
    if (reactionEl) {
      message.reactions = reactionEl.textContent?.trim();
    }

    // --- TASK 7: Structured quoted messages ---
    const quotedEl = msgEl.querySelector('[data-testid="quoted-message"]');
    if (quotedEl) {
      const quotedSenderEl = quotedEl.querySelector('[data-testid="quoted-message-author"]')
                          || quotedEl.querySelector('span[dir="auto"]:first-child');
      const quotedTextEl = quotedEl.querySelector('[data-testid="quoted-message-text"]')
                        || quotedEl.querySelector('span.selectable-text')
                        || quotedEl.querySelector('span[dir]');

      // Detect quoted media type
      let quotedType = 'text';
      if (quotedEl.querySelector('img')) quotedType = 'image';
      else if (quotedEl.querySelector('[data-testid*="video"]')) quotedType = 'video';
      else if (quotedEl.querySelector('[data-testid*="ptt"], [data-testid*="audio"]')) quotedType = 'audio';
      else if (quotedEl.querySelector('[data-testid*="document"]')) quotedType = 'document';

      const quotedSender = quotedSenderEl?.textContent?.trim() || '';
      let quotedText = '';

      // Get the quoted text, but avoid re-grabbing the sender
      if (quotedTextEl && quotedTextEl !== quotedSenderEl) {
        quotedText = quotedTextEl.textContent?.trim().substring(0, 500) || '';
      } else {
        // Fallback: get all text minus the sender
        const fullText = quotedEl.textContent?.trim() || '';
        quotedText = quotedSender ? fullText.replace(quotedSender, '').trim() : fullText;
        quotedText = quotedText.substring(0, 500);
      }

      message.quotedMessage = {
        sender: quotedSender,
        text: quotedText,
        type: quotedType
      };
    }

    // Post-processing: for media messages where sender is empty and content is
    // a short non-placeholder word, the "content" is likely the leaked sender name.
    // Move it to sender and clear content.
    if (message.media && !message.sender && message.content
        && !message.content.startsWith('[')
        && message.content.length <= 40
        && !/\s{2,}/.test(message.content)) {
      log(`Sender leak fix: moving content "${message.content}" to sender for media msg`);
      message.sender = message.content;
      message.content = `[${message.media.type.charAt(0).toUpperCase() + message.media.type.slice(1)}]`;
    }

    // Clean sender element attribute before returning (avoid DOM pollution across runs)
    if (senderEl) {
      senderEl.removeAttribute('data-wa-sender-extracted');
    }

    // Return only if we have content
    if (message.content || message.media || message.isDeleted) {
      return message;
    }

    return null;
  }

  // Extract media
  async function extractMedia(messages) {
    const mediaMessages = messages.filter(m => m.media && m.media.src);

    for (let i = 0; i < mediaMessages.length; i++) {
      const msg = mediaMessages[i];
      const filename = `${msg.media.type}_${i + 1}.${getExtension(msg.media.type)}`;

      // BUG-04: Use pre-fetched inline base64 if available (captured while
      // the element was visible in the viewport during extraction).
      if (msg.media._inlineBase64) {
        exportData.media.push({
          messageId: msg.id,
          type: msg.media.type,
          filename: filename,
          base64: msg.media._inlineBase64
        });
        msg.media.exportedFilename = filename;
        msg.media.src = filename;
        delete msg.media._inlineBase64;
      } else if (msg.media?.src?.startsWith('blob:')) {
        try {
          const response = await fetch(msg.media.src);
          const blob = await response.blob();
          const base64 = await blobToBase64(blob);

          exportData.media.push({
            messageId: msg.id,
            type: msg.media.type,
            filename: filename,
            base64: base64
          });

          msg.media.exportedFilename = filename;
          msg.media.src = filename;
        } catch (error) {
          log('Media extraction failed (blob likely revoked):', error);
        }
      }

      sendProgress(80 + (i / mediaMessages.length * 10), 'Processing media...', `${i + 1} of ${mediaMessages.length}`);
    }
  }

  // Build an array of {path, content/base64} objects for all export files.
  // Used when the popup writes files directly to a user-chosen directory
  // via the File System Access API (options.returnData = true).
  function buildExportFiles(chatName, format) {
    const safeName = sanitizeFilename(chatName);
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const dt = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    const baseDir = `${safeName}/${dt}`;
    const files = [];

    if (format === 'txt') {
      files.push({
        path: `${baseDir}/${safeName}.txt`,
        content: generateTXT(exportData.messages)
      });
    } else if (format === 'csv') {
      files.push({
        path: `${baseDir}/${safeName}.csv`,
        content: '\uFEFF' + generateCSV(exportData.messages)
      });
    } else {
      files.push({
        path: `${baseDir}/${safeName}.json`,
        content: JSON.stringify({
          exportInfo: {
            chatName: exportData.chatName,
            exportDate: exportData.exportDate,
            totalMessages: exportData.messages.length,
            totalMedia: exportData.media.length,
            exportedBy: 'WhatsApp Chat Exporter Extension v1.5'
          },
          messages: exportData.messages,
          systemMessages: exportData.systemMessages,
          mediaFiles: exportData.media.map(m => ({
            messageId: m.messageId,
            type: m.type,
            filename: m.filename
          }))
        }, null, 2)
      });
    }

    for (const media of exportData.media) {
      if (media.base64) {
        files.push({
          path: `${baseDir}/media/${media.filename}`,
          base64: media.base64
        });
      }
    }

    return files;
  }

  // Download export — supports JSON and CSV formats.
  // Folder structure: {chatName}/{YYYYMMDD_HHMMSS}/{chatName}.json
  //                   {chatName}/{YYYYMMDD_HHMMSS}/media/{file}
  async function downloadExport(chatName, format) {
    const safeName = sanitizeFilename(chatName);
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const datetimeStr = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    const baseDir = `${safeName}/${datetimeStr}`;

    if (format === 'txt') {
      // IMP-09: WhatsApp-native TXT format
      const txtContent = generateTXT(exportData.messages);
      downloadFile(txtContent, `${baseDir}/${safeName}.txt`, 'text/plain;charset=utf-8');
    } else if (format === 'csv') {
      const csvContent = generateCSV(exportData.messages);
      const csvWithBom = '\uFEFF' + csvContent;
      downloadFile(csvWithBom, `${baseDir}/${safeName}.csv`, 'text/csv;charset=utf-8');
    } else {
      // JSON export
      const jsonContent = JSON.stringify({
        exportInfo: {
          chatName: exportData.chatName,
          exportDate: exportData.exportDate,
          totalMessages: exportData.messages.length,
          totalMedia: exportData.media.length,
          exportedBy: 'WhatsApp Chat Exporter Extension v1.5'
        },
        messages: exportData.messages,
        systemMessages: exportData.systemMessages,
        mediaFiles: exportData.media.map(m => ({
          messageId: m.messageId,
          type: m.type,
          filename: m.filename
        }))
      }, null, 2);

      downloadFile(jsonContent, `${baseDir}/${safeName}.json`, 'application/json;charset=utf-8');
    }

    // Download media files
    for (const media of exportData.media) {
      if (media.base64) {
        try {
          const binaryData = atob(media.base64.split(',')[1] || media.base64);
          const bytes = new Uint8Array(binaryData.length);
          for (let i = 0; i < binaryData.length; i++) {
            bytes[i] = binaryData.charCodeAt(i);
          }
          const blob = new Blob([bytes], { type: getMimeType(media.type) });
          const url = URL.createObjectURL(blob);

          chrome.runtime.sendMessage({
            action: 'download',
            url: url,
            filename: `${baseDir}/media/${media.filename}`
          });
        } catch (e) {
          log('Media download failed:', e);
        }
      }
    }
  }

  // IMP-09: Generate TXT in WhatsApp-native export format
  function generateTXT(messages) {
    const lines = [];
    for (const msg of messages) {
      const date = msg.date || '';
      const time = msg.timestamp || '';
      const sender = msg.sender || 'Unknown';
      const content = msg.content || '';
      if (date && time) {
        lines.push(`[${date}, ${time}] ${sender}: ${content}`);
      } else if (time) {
        lines.push(`[${time}] ${sender}: ${content}`);
      } else {
        lines.push(`${sender}: ${content}`);
      }
    }
    return lines.join('\n');
  }

  // --- TASK 8: Generate CSV from messages ---
  function generateCSV(messages) {
    const escapeCSV = (val) => {
      if (val == null) return '';
      const str = String(val);
      if (str.includes('"') || str.includes(',') || str.includes('\n') || str.includes('\r')) {
        return '"' + str.replace(/"/g, '""') + '"';
      }
      return str;
    };

    const headers = ['date', 'time', 'dateTime', 'sender', 'type', 'content', 'media_filename', 'is_outgoing'];
    const rows = [headers.join(',')];

    for (const msg of messages) {
      const row = [
        escapeCSV(msg.date || ''),
        escapeCSV(msg.timestamp || ''),
        escapeCSV(msg.dateTime || ''),
        escapeCSV(msg.sender || ''),
        escapeCSV(msg.type || ''),
        escapeCSV(msg.content || ''),
        escapeCSV(msg.media?.exportedFilename || msg.media?.src || ''),
        escapeCSV(msg.isOutgoing ? 'true' : 'false')
      ];
      rows.push(row.join(','));
    }

    return rows.join('\r\n');
  }

  // Utilities
  function downloadFile(content, filename, mimeType) {
    // --- TASK 3: Ensure UTF-8 Blob encoding ---
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    chrome.runtime.sendMessage({ action: 'download', url: url, filename: filename });
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  function getExtension(type) {
    return { 'image': 'jpg', 'video': 'mp4', 'audio': 'ogg', 'document': 'pdf' }[type] || 'bin';
  }

  function getMimeType(type) {
    return { 'image': 'image/jpeg', 'video': 'video/mp4', 'audio': 'audio/ogg', 'document': 'application/pdf' }[type] || 'application/octet-stream';
  }

  function generateId() {
    return 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  // IMP-10: Robust filename sanitization
  function sanitizeFilename(name) {
    let safe = (name || '').replace(/[^a-z0-9\s._-]/gi, '_');
    // Handle names that become all-underscores or empty
    if (!safe.replace(/[_\s]/g, '')) safe = 'chat_export';
    // Remove leading/trailing dots, spaces, underscores
    safe = safe.replace(/^[\s._]+|[\s._]+$/g, '');
    // Windows reserved filenames
    if (/^(CON|PRN|AUX|NUL|COM\d|LPT\d)$/i.test(safe)) safe = '_' + safe;
    safe = safe.substring(0, 50).replace(/[\s.]+$/, '');
    return safe || 'chat_export';
  }

  // IMP-07: Validate export output before download
  function validateExport(messages) {
    const warnings = [];
    if (messages.length === 0) {
      warnings.push('No messages found. WhatsApp\'s page structure may have changed.');
      return { valid: false, warnings };
    }
    const withSender = messages.filter(m => m.sender && m.sender.length > 0).length;
    const senderPct = Math.round((withSender / messages.length) * 100);
    if (senderPct < 80) {
      warnings.push(`Only ${senderPct}% of messages have an identified sender.`);
    }
    const withDate = messages.filter(m => m.date).length;
    const datePct = Math.round((withDate / messages.length) * 100);
    if (datePct < 50) {
      warnings.push(`Only ${datePct}% of messages have a date.`);
    }
    return { valid: warnings.length === 0, warnings };
  }

  // IMP-03: DJB2 hash of sender + timestamp + content prefix for dedup
  function messageContentHash(msg) {
    const key = `${msg.sender || ''}|${msg.timestamp || ''}|${(msg.content || '').substring(0, 100)}`;
    let hash = 5381;
    for (let i = 0; i < key.length; i++) {
      hash = ((hash << 5) + hash) + key.charCodeAt(i);
      hash |= 0;
    }
    return hash.toString(36);
  }

  // IMP-02: sleep with optional AbortSignal support
  function sleep(ms, signal) {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) { reject(new DOMException('Aborted', 'AbortError')); return; }
      const timer = setTimeout(resolve, ms);
      if (signal) {
        const onAbort = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
        signal.addEventListener('abort', onAbort, { once: true });
      }
    });
  }

  // IMP-06: Adaptive sleep — resolves early once DOM mutations settle
  // (no new mutations for 300 ms), with a hard cap of maxMs.
  function adaptiveSleep(maxMs, signal) {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) { reject(new DOMException('Aborted', 'AbortError')); return; }
      const panel = document.querySelector(SELECTORS.mainPanel);
      if (!panel) return sleep(maxMs, signal).then(resolve, reject);

      let resolved = false;
      let settleTimer = null;
      let maxTimer = null;

      const done = () => {
        if (resolved) return;
        resolved = true;
        observer.disconnect();
        clearTimeout(settleTimer);
        clearTimeout(maxTimer);
        resolve();
      };

      const observer = new MutationObserver(() => {
        clearTimeout(settleTimer);
        settleTimer = setTimeout(done, 300);
      });
      observer.observe(panel, { childList: true, subtree: true });

      // If no mutations at all, settle after 300 ms
      settleTimer = setTimeout(done, 300);
      // Hard cap
      maxTimer = setTimeout(done, maxMs);

      if (signal) {
        const onAbort = () => {
          if (!resolved) {
            resolved = true;
            observer.disconnect();
            clearTimeout(settleTimer);
            clearTimeout(maxTimer);
            reject(new DOMException('Aborted', 'AbortError'));
          }
        };
        signal.addEventListener('abort', onAbort, { once: true });
      }
    });
  }

  // BUG-10: Include sessionId so the popup can filter stale progress messages
  let _currentSessionId = null;

  function sendProgress(percent, label, detail) {
    const msg = { action: 'progressUpdate', percent, label, detail };
    if (_currentSessionId) msg.sessionId = _currentSessionId;
    chrome.runtime.sendMessage(msg)
      .catch(() => {}); // Suppress rejection when popup is not open
  }

  log('WhatsApp Chat Exporter loaded (v1.5 - audit fixes)');
})();
