#!/usr/bin/python
# -*- coding: utf-8 -*-

 # @author Olof Mogren
 #
 # email: olof@mogren.one
 #
 # 
 # Base class for submodular summarization.
 # 
 # Providing some base functions.
 #
 # Released under the GNU GPL.
 # 
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 #
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program.  If not, see <http://www.gnu.org/licenses/>.
 #

import time, urlparse, webbrowser, threading, sys, re
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from multiprocessing import Process
from threading import Thread
import multsum, backend_worker, backend_client



HOST_NAME         = ''
PORT_NUMBER_FIRST = 9291
PORT_NUMBER_LAST  = 9299

use_w2v_similarity = True
show_exit_button   = False

BASE_DOCUMENT_PREFIX = '''<!DOCTYPE html>
<html>
<head>
<title>MULTSUM GUI</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<style>
.divleft, .divright {
    display: block;
    width: 100%;
}

@media only screen and (min-width: 481px) {
    .divleft, .divright {
        display: inline-block;
        vertical-align: top;
        width: 49%;
        /*float: left;*/
    }
}

</style>
<script>

function request(button_id)
{
  var requestPath = '/get_summary';
  if( button_id == 'exit_button')
  {
    requestPath = '/exit';
    document.getElementById("output_p").innerHTML = 'Exiting... Please wait.';
  }
  else if(button_id == 'submit_button')
  {
    requestPath = '/get_summary';
    document.getElementById("output_p").innerHTML = 'Computing summary... Please wait.';
  }
  var xhr = new XMLHttpRequest();
  xhr.open("POST", requestPath, true);
  var params = "input_text=" + encodeURIComponent(document.getElementById("input_text").value);
  params += "&summary_length=" + encodeURIComponent(document.getElementById("summary_length").value);
  params += "&tfidf=" + encodeURIComponent(document.getElementById("tfidf").checked);
  params += "&sentiment=" + encodeURIComponent(document.getElementById("sentiment").checked);
  params += "&w2v=" + encodeURIComponent(document.getElementById("w2v").checked);

  xhr.onload = function (e) {
    if (xhr.readyState === 4) {
      if (xhr.status === 200) {
        console.log(xhr.responseText);
        document.getElementById('output_legend').innerHTML = 'Output';
        document.getElementById('output_p').innerHTML = xhr.responseText;
        if( button_id == 'exit_button')
        {
          document.getElementById('output_legend').innerHTML = 'Status';
          document.getElementById('submit_button').disabled = true;
          document.getElementById('exit_button').disabled = true;
        }
      } else {
        document.getElementById('output_legend').innerHTML = 'Status';
        document.getElementById("output_p").innerHTML = xhr.statusText;
        console.error(xhr.statusText);
      }
    }
  };
  xhr.onerror = function (e) {
    document.getElementById('output_legend').innerHTML = 'Status';
    document.getElementById("output_p").innerHTML = xhr.statusText;
    console.error(xhr.statusText);
  };
  xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
  xhr.send(params);
}
function leaving(e)
{
  if(!e) e = window.event;
    //e.cancelBubble is supported by IE - this will kill the bubbling process.
    e.cancelBubble = true;
    e.returnValue = 'You sure you want to leave the demo?'; //This is displayed on the dialog
    
    //e.stopPropagation works in Firefox.
    if (e.stopPropagation) {
      e.stopPropagation();
      e.preventDefault();
    }
}
//window.onbeforeunload = leaving

function checkW2V()
{
  var requestPath = '/backend_status';
  var xhr = new XMLHttpRequest();
  xhr.open("GET", requestPath, true);
  xhr.onload = function (e) {
    if (xhr.readyState === 4) {
      if (xhr.status === 200) {
        console.log(xhr.responseText);
        if(xhr.responseText == 'OK')
        {
          document.getElementById('w2v_label').style.color = "#000";
          document.getElementById('w2v').disabled = false;
          document.getElementById('w2v').checked  = true;
        }
        else
        {
          window.setTimeout(checkW2V, 2000)
        }
      } else {
        console.error(xhr.statusText);
      }
    }
  };
  xhr.onerror = function (e) {
    console.error(xhr.statusText);
  };
  xhr.send();
}
</script>

</head>
<body style="font-family: arial;" onload="checkW2V()">
<div style="min-width: 640px; max-width: 1200px; margin-left: auto; margin-right: auto ;">
  <h1 id="header" style="font-family: arial; text-align: center;">MULTSUM</h1>

'''
BASE_DOCUMENT_FORM_1 = '''
  <div class="divleft">
    <form action="/ajaxless" method="POST">
      <fieldset>
      <legend style="font-family: arial;">Input text to summarize</legend>
      <textarea style="width: 100%; height: 400px;" name="input_text" id="input_text"></textarea>
      <div style="width: 100%;" id="div_settings">
        <select name="summary_length" id="summary_length">
          <option value="50">50</option>
          <option value="100" selected="selected">100</option>
          <option value="200">200</option>
          <option value="400">400</option>
        </select>
        <label for="summary_length">Summary length (words)</label>
        <br />
        <input type="checkbox" id="tfidf" name="tfidf" value="tfidf" checked="checked" />
        <label for="tfidf">LinTFIDF</label>
        <br />
        <input type="checkbox" id="sentiment" name="sentiment" value="sentiment" checked="checked" />
        <label for="sentiment">Sentiment</label>
        <br />
        <input type="checkbox" id="w2v" name="w2v" value="w2v" disabled="disabled" />
        <label for="w2v" id="w2v_label" style="color: #ccc;">Word2Vec</label>
      </div>
    <br />
'''
BASE_DOCUMENT_FORM_EXIT_BUTTON = '''<button id="exit_button" onclick="request('exit_button'); return false;">Exit</button>
'''

BASE_DOCUMENT_FORM_2 = '''      <button id="submit_button" onclick="request('submit_button'); return false;" style="float: right;">Summarize</button>
      </fieldset>
    </form>
  </div>
  <div  class="divright">
    <form action="/" method="POST">
      <fieldset>
        <legend id="output_legend" style="font-family: arial;">&nbsp;</legend>
        <!--<textarea style="width: 100%; height: 100px;" id="output_area" readonly="readonly"></textarea>//-->
        <div id="output_div"><p id="output_p"></p></div>
      </fieldset>
    </form>
  </div>
'''

BASE_DOCUMENT_SUFFIX = '''


<p style="font-size: 10px; font-family: arial;">For more information, please visit <a href="http://mogren.one/" target="_blank">mogren.one</a>.</p>

<p style="font-size: 10px; font-family: arial;">Relevant publications:
<br />
"Extractive Summarization by Aggregating Multiple Similarities", Olof Mogren, Mikael Kågebäck, Devdatt Dubhashi, RANLP 2015.
<br />
"Extractive Summarization using Continuous Vector Space Models", Mikael Kågebäck, Olof Mogren, Nina Tahmasebi, Devdatt Dubhashi, CVSC@EACL 2014.</p>

<br />
<br />
<a href="http://www.cse.chalmers.se/research/lab/"><img style="border-width: 0px; float: right;" src="'''+\
'''data:image/jpeg;base64,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'''+\
'''" alt="Chalmers University of Technology" /></a>
<br />&nbsp;
</div>
</body>
</html>
'''


def convert(s):
  try:
    return s.group(0).encode('latin1').decode('utf8')
  except:
    return s.group(0)

def convert_all(s):
  return re.sub(r'[\x80-\xFF]+', convert, s)

class ThreadedHTTPServer(HTTPServer):
  def process_request(self, request, client_address):
    thread = Thread(target=self.__new_request, args=(self.RequestHandlerClass, request, client_address, self))
    thread.start()
  def __new_request(self, handlerClass, request, address, server):
    handlerClass(request, address, server)
    self.shutdown_request(request)


class MyHandler(BaseHTTPRequestHandler):
  def do_HEAD(s):
    s.send_response(200)
    s.send_header("Content-type", "text/html")
    s.end_headers()
  def do_GET(s):
    MyHandler.respond(s)
  def do_POST(s):
    MyHandler.respond(s)

  def respond(s):
    global show_exit_button
    """Respond to a POST request."""
    print 'path: '+s.path
    if s.path == '/exit':
      if not show_exit_button:
        print 'Ignoring exit request. show_exit_button=False.'
      else:
        print 'Got exit command'
        s.send_response(200)
        s.send_header("Content-type", "text/plain;charset=UTF-8")
        s.end_headers()
        s.wfile.write('System is exiting.')
        assassin = Thread(target=s.server.shutdown)
        assassin.daemon = True
        assassin.start()
    elif s.path == '/get_summary':
      print 'Got summarize command'
      # Extract and print the contents of the POST
      length = int(s.headers['Content-Length'])
      err = None
      try:
        post_data = urlparse.parse_qs(s.rfile.read(length).decode('utf-8'))
        input_text     = post_data['input_text'][0]
        summary_length = int(post_data['summary_length'][0])
        print 'length: '+str(summary_length)
        tfidf          = (post_data['tfidf'][0] == "tfidf")
        sentiment      = (post_data['sentiment'][0] == "sentiment")
        w2v            = (post_data['w2v'][0] == "w2v")
        for key, value in post_data.iteritems():
          print "%s=%s" % (key, value)
          #if key == 'input_text': input_text = value
        #print u'input_text: '+convert_all(input_text)
        input_text = convert_all(input_text)
        #print input_text.encode('latin1').decode('utf-8')
      except:
        err = 'Failure while processing input. Please make sure that you input valid text.'
      try:
        summary = multsum.summarize_strings([[input_text]], length=summary_length, use_tfidf_similarity=tfidf, use_sentiment_similarity=sentiment, use_w2v_similarity=w2v, split_sentences=True, w2v_backend=True)
      except:
        err = 'Failed to run the summarization subsystem. Internal error. Make sure that your input is valid.'

      s.send_response(200)
      s.send_header("Content-type", "text/plain;charset=UTF-8")
      s.end_headers()
      if not err:
        s.wfile.write(summary.encode('utf-8'))
      else:
        s.wfile.write(err)
    elif s.path == '/':
      print 'Got root request.'
      s.send_response(200)
      s.send_header("Content-type", "text/html;charset=UTF-8")
      s.end_headers()
      s.wfile.write(BASE_DOCUMENT_PREFIX)
      s.wfile.write(BASE_DOCUMENT_FORM_1)
      if show_exit_button:
        s.wfile.write(BASE_DOCUMENT_FORM_EXIT_BUTTON)
      s.wfile.write(BASE_DOCUMENT_FORM_2)
      s.wfile.write(BASE_DOCUMENT_SUFFIX)
    elif s.path == '/backend_status':
      s.send_response(200)
      s.send_header("Content-type", "text/plain;charset=UTF-8")
      s.end_headers()
      if use_w2v_similarity and backend_client.backend_check():
        s.wfile.write('OK')
      else:
        s.wfile.write('NOK')
    else:
      print 'Got other request (404).'
      s.send_response(404)
      s.send_header("Content-type", "text/html;charset=UTF-8")
      s.end_headers()
      s.wfile.write('''<html><head><title>404: Page not found.</title></head><body><h1>404: Page not found</h1></body></html>
      ''')

def run_server(port_number=PORT_NUMBER_FIRST):
  server_class = ThreadedHTTPServer
  httpd = server_class((HOST_NAME, port_number), MyHandler)
    
  print time.asctime(), "Server Starts - %s:%s" % (HOST_NAME, port_number)
  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    pass
  httpd.server_close()
  print time.asctime(), "Server Stops - %s:%s" % (HOST_NAME, port_number)


def run_browser():
  from gi.repository import Gtk, Gdk, WebKit
  Gtk.init(sys.argv)
  
  #self.ebview.go_back()
  #self.webview.goforward()

  win = Gtk.Window()
  scrolled_window = Gtk.ScrolledWindow()
  webview = WebKit.WebView()
  scrolled_window.add(webview)
  win.add(scrolled_window)
  win.set_title('MULTSUM')

  win.connect("delete-event", Gtk.main_quit)
  #self.webview.show()
  #scrolled_window.show()
  win.show_all()
  webview.load_uri('http://localhost:%s/'%(str(port_number)))
  win.resize(1000, 800)

  #browser = Browser()
  Gtk.main()


if __name__ == '__main__':
  print multsum.WELCOME_MESSAGE
  import signal
  signal.signal(signal.SIGINT, signal.SIG_DFL)
  
  launch_browser = True
  w2v_vector_file = None

  skip = False
  for i in range(1,len(sys.argv)):
    if skip:
      skip = False
      continue
    if sys.argv[i] == '--no-browser':
      launch_browser = False
    elif sys.argv[i] == '--no-w2v':
      use_w2v_similarity = False
    elif sys.argv[i] == '--w2v-file':
      w2v_vector_file = sys.argv[i+1]
      skip = True
            
    #elif sys.argv[i] == '--no-exit-button':
    #  show_exit_button = False

  #p_model = Process(target=init_model)
  #p_model.start()
  for port_number in range(PORT_NUMBER_FIRST,PORT_NUMBER_LAST):
    try:
      p = Process(target=run_server, kwargs={'port_number': port_number})
      p.start()
      break
    except socket.error, e:
      print('Failed to start http server at port %d.'%(port_number))

  backend_started = False
  if use_w2v_similarity:
    if not backend_client.backend_check() and not backend_client.backend_is_initializing():
      p_backend = Process(target=backend_worker.run_backend, args={'w2v_vector_file': w2v_vector_file})
      p_backend.start()
      backend_started = True
  if launch_browser:
    try:
      run_browser()
      try:
        p.terminate()
        if backend_started:
          print('Sending exit command to w2v worker.')
          backend_client.backend_exit()
          p_backend.terminate()
      except:
        print 'Failed to kill server. Please do so manually.'
    except:
      print 'Failed internal browser. Will try to launch system broswer.'
      try:
        webbrowser.open('http://localhost:%d/'%(port_number))
      except:
        print 'Failed to start web interface. Start a broswer, and point it to http://localhost:%d/.'%(port_number)
  else:
    print 'Open a browser and point it to http://localhost:%d/.'%(port_number)
  
  print('Waiting web server to finish.')
  p.join()
  print('Done.')
  #p_model.join()

