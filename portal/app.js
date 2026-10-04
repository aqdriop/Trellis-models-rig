/* Mejora progresiva del portal: datos del ZIP desde downloads/manifest.json, copiar el SHA-256
   y un aviso al iniciar la descarga. Sin dependencias; la página y el botón funcionan sin JS. */
(function () {
  'use strict';

  var MANIFEST_URL = 'downloads/manifest.json';
  var DESCRIPTIONS = {
    'TRELLIS_AutoRig_Colab.ipynb': 'el cuaderno de Colab',
    'LEEME.md': 'guía de uso y solución de problemas'
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, text) {
    var el = byId(id);
    if (el) {
      el.textContent = text;
    }
  }

  function formatSize(bytes) {
    var kb = bytes / 1024;
    if (kb < 1024) {
      return new Intl.NumberFormat('es', { maximumFractionDigits: 1 }).format(kb) + ' KB';
    }
    return new Intl.NumberFormat('es', { maximumFractionDigits: 1 }).format(kb / 1024) + ' MB';
  }

  function formatDate(iso) {
    var parts = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || '');
    if (!parts) {
      return iso || '—';
    }
    var date = new Date(Date.UTC(+parts[1], +parts[2] - 1, +parts[3]));
    return new Intl.DateTimeFormat('es', { dateStyle: 'long', timeZone: 'UTC' }).format(date);
  }

  function startsWithOneOf(value, prefixes) {
    return typeof value === 'string' && prefixes.some(function (p) { return value.indexOf(p) === 0; });
  }

  function renderContents(contents) {
    var list = byId('zip-contents');
    if (!list || !Array.isArray(contents) || !contents.length) {
      return;
    }
    list.textContent = '';
    contents.forEach(function (entry) {
      var name = String(entry.path || '').split('/').pop();
      if (!name) {
        return;
      }
      var item = document.createElement('li');
      var code = document.createElement('code');
      code.textContent = name;
      var desc = document.createElement('span');
      desc.className = 'desc';
      desc.textContent = DESCRIPTIONS[name] || formatSize(Number(entry.bytes) || 0);
      item.appendChild(code);
      item.appendChild(document.createTextNode(' '));
      item.appendChild(desc);
      list.appendChild(item);
    });
  }

  function applyManifest(manifest) {
    var size = formatSize(Number(manifest.bytes) || 0);
    var version = 'v' + String(manifest.version || '').replace(/^v/, '');

    var metaSize = byId('meta-size');
    var metaVersion = byId('meta-version');
    if (metaSize) {
      metaSize.textContent = size;
      metaSize.hidden = false;
    }
    if (metaVersion) {
      metaVersion.textContent = 'versión ' + version;
      metaVersion.hidden = false;
    }

    setText('fact-size', size);
    setText('fact-version', version);
    setText('fact-updated', formatDate(manifest.updated));
    setText('sha-value', String(manifest.sha256 || 'no disponible'));
    renderContents(manifest.contents);

    var colab = byId('colab-link');
    if (colab && startsWithOneOf(manifest.colab_url, ['https://colab.research.google.com/'])) {
      colab.href = manifest.colab_url;
    }
    var release = byId('release-link');
    if (release && startsWithOneOf(manifest.download_url, ['https://github.com/'])) {
      release.href = manifest.download_url;
    }

    var copy = byId('copy-sha');
    if (copy && manifest.sha256) {
      copy.hidden = false;
    }
  }

  function copyWithSelection(text) {
    return new Promise(function (resolve, reject) {
      var area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', '');
      area.className = 'visually-hidden';
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand('copy') ? resolve() : reject(new Error('copy'));
      } catch (error) {
        reject(error);
      } finally {
        document.body.removeChild(area);
      }
    });
  }

  function copyToClipboard(text) {
    // La API moderna puede rechazarse (permisos, iframes): se prueba el método clásico después.
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).catch(function () {
        return copyWithSelection(text);
      });
    }
    return copyWithSelection(text);
  }

  function wireCopyButton() {
    var button = byId('copy-sha');
    var label = byId('copy-label');
    if (!button || !label) {
      return;
    }
    var timer = null;
    button.addEventListener('click', function () {
      var value = (byId('sha-value') || {}).textContent || '';
      copyToClipboard(value).then(
        function () { label.textContent = 'Copiado'; },
        function () { label.textContent = 'Selecciona y copia'; }
      ).then(function () {
        window.clearTimeout(timer);
        timer = window.setTimeout(function () { label.textContent = 'Copiar'; }, 2200);
      });
    });
  }

  function wireDownloadNotice() {
    var button = byId('download-btn');
    if (!button) {
      return;
    }
    button.addEventListener('click', function () {
      setText(
        'download-status',
        'Descarga iniciada. Siguiente paso: descomprime y súbelo a Colab (Archivo → Subir cuaderno).'
      );
    });
  }

  wireCopyButton();
  wireDownloadNotice();

  if (window.fetch) {
    fetch(MANIFEST_URL, { cache: 'no-cache' })
      .then(function (response) {
        if (!response.ok) {
          throw new Error('HTTP ' + response.status);
        }
        return response.json();
      })
      .then(applyManifest)
      .catch(function () {
        setText('sha-value', 'No se pudo cargar la huella; consulta downloads/manifest.json.');
      });
  }
}());
