/* global fetch alert $ */

/**
 * Send a message to the server.
 * @param {string} url - Send the url to python backend.
 * @param {string} data - The literal text content.
 * @param {*} wrapper - The function preprocessing input data.
 */
async function postData (url, data, wrapper) {
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-type': 'application/json'
      },
      body: JSON.stringify({
        source: wrapper(data)
      })
    })
    return await response.json()
  } catch (e) {
    console.error(e)
    return { message: e }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  console.log('DOM fully loaded and parsed')
  $('#history').DataTable()
})

document.addEventListener('click', e => {
  // Send text input to the backend.
  if (e.target.id === 'translate') {
    console.debug('POST literal context to server side.')
    const docInput = document.getElementById('doc_input').value
    // if types in chinese or not
    if (chTest(docInput)) {
      document.getElementById('source_label').innerHTML = 'Please use English.'
      alert('Please type words in English.')
      return
    }
    const tokens = docInput.split(' ')
    // type enough length text
    if (lengthTest(tokens)) {
      document.getElementById('source_label').innerHTML =
        'Please fill in the textarea with more than 5 tokens.'
      alert('Please fill in the textarea with more than 5 tokens.')
      return
    }
    document.getElementById('summary_output').innerHTML = ''
    document.getElementById('source_label').innerHTML = 'Loading ...'
    $('.loader').show()

    postData(('/run_decode'), (tokens), (updateText))
      .then(data => {
        const hypWords = docInput.replace('\n', ' ').split(' ')
        const refSet = new Set(data.final.replace('\n', ' ').split(' '))
        const sum = data.final.replace('\n', '<br>')
        // const ind = data["sentNum"]
        $('.loader').hide()
        document.getElementById('summary_output').innerHTML = sum
        document.getElementById('source_label').innerHTML = 'Inpur your document'

        for (const word of refSet) {
          for (let i = 0; i < hypWords.length; i++) {
            if (hypWords[i].toLowerCase() === word.toLowerCase()) {
              hypWords[i] =
                '<span style="background-color: yellow">' +
                word +
                '</span>'
            }
          }
        }
        let ans = ''
        for (let i = 0; i < hypWords.length; i++) {
          ans += hypWords[i] + ' '
        }
        document.getElementById('summary_output').innerHTML +=
          '<hr><b>OriginalText</b><br><br><div>' + ans + '</div>'
      })
      .catch(err => {
        console.error(err)
        alert(JSON.stringify(err.message))
      })
  } else if (e.target.matches('.dataset')) {
    console.debug('Open the link of dataset.')
    window.open('https://github.com/ChenRocks/cnn-dailymail', '_blank')
  }
})

const historicEventBtn = document.getElementById('history')
historicEventBtn.addEventListener('dblclick', e => {
  const tr = e.target.closest('tr')
  const answer = window.confirm('Are you sure to delete this record?')
  if (answer) {
    console.debug('delete a record')
    tr.remove()
  }
})

function chTest (tokens) {
  for (let i = 0; i < tokens.length; i++) {
    if (tokens.charCodeAt(i) >= 0x4e00 && tokens.charCodeAt(i) <= 0x9fa5) {
      return true
    }
  }
  return false
}

function lengthTest (tokens) {
  if (tokens.length <= 5) return true
  else return false
}
/**
 * Update original text by filling in spaces0
 * @param {string} tokens - Convert sep sybol to space.
 */
function updateText (tokens) {
  let sep = false
  let newDocInput = ''

  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].includes('\n')) {
      if (!sep) newDocInput += '\n'
      else newDocInput += tokens[i].replace('\n', '') + ' '
      sep = true
    } else {
      newDocInput += tokens[i] + ' '
      sep = false
    }
  }
  return newDocInput.substring(0, newDocInput.length - 1)
}

function resizeIframe (obj) {
  obj.style.height =
    obj.contentWindow.document.documentElement.scrollHeight + 'px'
}
