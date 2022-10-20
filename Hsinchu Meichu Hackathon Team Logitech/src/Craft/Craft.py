'''
 Copyright 2018 Logitech Inc.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files(the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
'''

import wx
import websocket
import threading
import os
import platform
import json
import uuid
import sys
import math
from copy import deepcopy
from time import sleep
from json import JSONEncoder
from uuid import UUID

import pyautogui

glist=[]
crownObjectList = []
toolObject = []
sessionId=""


JSONEncoder_olddefault = JSONEncoder.default

# special encoding for supporting UUID in json payload
def JSONEncoder_newdefault(self, o):
    if isinstance(o, UUID): return str(o)
    return JSONEncoder_olddefault(self, o)

JSONEncoder.default = JSONEncoder_newdefault

class CraftClient(object):

    def __init__(self):
        self.executableName=""
        self.manifestPath=""
        self.callback=""
    
    def on_message(self,ws, message):
        print("on_message called...")
        # craft events come in as json objects
        craftEventObj = json.loads(message)
        wx.CallAfter(self.wrapperUpdateUI, craftEventObj)

    def on_close(self,ws):
        print("### closed ###")

    def on_open(self,ws):
        print("on_open called...")
        uid = "6202f2fb-834c-4393-a95f-f5051171e3ec"
        pid = os.getpid()

        connectMessage = {
            "message_type": "register",
            "plugin_guid": uid,
            "PID": pid,
            "execName": self.executableName,
            "manifestPath": self.manifestPath,
            "application_version": "0.0.0"
        }

        regMsg =  json.dumps(connectMessage)
        ws.send(regMsg.encode('utf8'))

    def connect(self, execName,manifestFilePath):
        print("connect called...")
        global ws
        self.executableName = execName
        self.manifestPath = manifestFilePath

        websocket.enableTrace(True)

        ws = websocket.WebSocketApp("ws://127.0.0.1:10134",
                                 on_open = self.on_open,
                                 on_message = self.on_message,
                                 on_close = self.on_close)


        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()

    def registerEventHandler(self, cb):
        self.callback = cb

    def report(self, event, value):
        toolId = event.get('task_options').get('current_tool')
        na = event.get('task_options').get('current_tool_option')

        response = {
            "message_type" : "tool_update",
            "session_id"   :  sessionId,
            "show_overlay" :  True,
            "tool_id"      :  toolId,
            "tool_options" :  [{
                "name"  : na,
                "value" : value
            }]
        };
        regMsg =  json.dumps(response)
        ws.send(regMsg.encode('utf8'))

    def wrapperUpdateUI(self,msg):
        global glist,sessionId
        totalDeltaValue=0
        totalRatchetDeltaValue=0
        count=0
        listCount=0
        global firstObject

        if(msg['message_type'] == "crown_turn_event"):
            glist.append(msg)
            listCount = len(glist)
            if listCount==0:
                return
            currentToolOption = glist[0]['task_options']['current_tool_option']
            print("+++currentToolOption = ",currentToolOption)
            print("listCount = ",listCount)
            firstObject = glist[0]
            for i in range(listCount):
                if currentToolOption == glist[i]['task_options']['current_tool_option']:
                    totalDeltaValue = totalDeltaValue = glist[i]['delta']
                    totalRatchetDeltaValue = totalRatchetDeltaValue + glist[i]['ratchet_delta']
                else:
                    break
                count += 1

            if listCount >= 0:
                glist.clear()
            print("totalDeltaValue = ",totalDeltaValue)
            print("firstObject = ",firstObject['message_type'])
            
            if firstObject['message_type'] == "deactivate_plugin":
                return

            try:
                if firstObject['message_type'] == "crown_turn_event":
                    print("turn event =====")
                    if firstObject['task_options']['current_tool'] == 'Slider':
                        print("\n","selected slider")
                        v = slider.GetValue()
                        tvalue = v + totalDeltaValue
                        if tvalue <= 0:
                            tvalue = 0
                        if tvalue >1000:
                            tvalue = 1000
                        slider.SetValue(tvalue)
                        print("report called....",tvalue,msg)
                        self.report(msg,tvalue)
                    elif firstObject['task_options']['current_tool'] == 'SpinCtrl':
                        print("\n","selected SpinCtrl")
                        v = spin.GetValue()
                        tvalue = v + totalDeltaValue
                        if tvalue <= 0:
                            tvalue = 0
                        if tvalue >1000:
                            tvalue = 1000
                        spin.SetValue(tvalue)
                        self.report(msg,tvalue)
                    elif firstObject['task_options']['current_tool'] == 'Gauge':
                        if firstObject['task_options']['current_tool_option'] == 'gauge':
                           print("\n","selected Gauge")
                           v = gauge.GetValue()
                           tvalue = v + totalDeltaValue
                           if tvalue <= 0:
                              tvalue = 0
                           if tvalue >500:
                              tvalue = 500
                           gauge.SetValue(tvalue)
                           self.report(msg,tvalue)
                        if firstObject['task_options']['current_tool_option'] == 'gaugeRatchet':
                           print("\n","selected gaugeRatchet")
                           v = gauge.GetValue()
                           tvalue = v + (totalRatchetDeltaValue * 10)
                           if tvalue <= 0:
                              tvalue = 0
                           if tvalue >500:
                              tvalue = 500
                           gauge.SetValue(tvalue)
                           self.report(msg,tvalue)

                    elif firstObject['task_options']['current_tool'] == 'ComboBox':
                        if totalRatchetDeltaValue != 0 :
                            print("\n","selected ComboBox")
                            v = combo.GetSelection()
                            width, height = pyautogui.size()
                            divider = 6
                            multi_wid = int(width/divider)
                            if multi_wid*6 != width:
                                divider = 8
                                multi_wid = int(width/divider)
                            x_now, y_now = pyautogui.position()
                            pyautogui.moveTo(v*multi_wid, y_now)
                            if (v == 1 and totalRatchetDeltaValue < 0) or (v == divider-1 and totalRatchetDeltaValue > 0):
                                widthto = 0
                            elif totalRatchetDeltaValue > 0:
                                widthto = multi_wid
                            else:
                                widthto = -multi_wid
                            pyautogui.drag(widthto, 0, 1, button='left')
                            #pyautogui.moveTo(x_now, y_now)
                            combo.SetSelection(v+widthto/multi_wid)
                            pyautogui.keyDown('alt')
                            pyautogui.press('tab')
                            pyautogui.keyUp('alt')
                            x_now, y_now = pyautogui.position()
                            print(x_now)
                            self.report(msg,tvalue)
                    elif firstObject['task_options']['current_tool'] == 'TextCtrl':
                        print("\n","selected TextCtrl")
                        v = txt.GetSize()
                        h = v.height + totalDeltaValue
                        w = v.width + totalDeltaValue
                        txt.SetSize(w,h)
                        self.report(msg,w)
                    elif firstObject['task_options']['current_tool'] == 'ListBox':
                        print("\n","selected ListBox")
                        v = lb.GetSelection()
                        v = v + totalRatchetDeltaValue
                        if v <= 0:
                            v = 0
                        if v > 999:
                            v = 999
                        lb.SetSelection(v)
                        lb.EnsureVisible(v)
                        self.report(msg,v)
            except ValueError:
                print("Error: update UI")



        elif (msg['message_type'] == "register_ack"):
            print("register_ack = ",msg['message_type'])
            sessionId = msg['session_id']
            print("Session Id = ",sessionId)

            defaultTool = "ComboBox"
            

            connectMessage = {
                "message_type": "tool_change",
                "session_id": sessionId,
                "tool_id": defaultTool
            }
            regMsg =  json.dumps(connectMessage)
            ws.send(regMsg.encode('utf8'))

class TestFrame(wx.Frame):

    def __init__(self, parent, id):
        global craft,slider,spin,gauge,combo,txt,lb
        width, height = pyautogui.size()
        wx.Frame.__init__(self, parent, id, "Craft Python SDK Sample", pos=(width,0),size=(1,1))
    
        panel = wx.Panel(self)

        l=[]
        for i in range(0, 10):
          l.append(str(i))

        combo = wx.ComboBox(panel, -1, "", pos=(0,0), size=(0,0), choices=l)
        combo.Bind(wx.EVT_SET_FOCUS, self.comboBoxFocus)
        combo.Bind(wx.EVT_LEFT_UP, self.comboBoxFocus)
        width, height = pyautogui.size()
        multi_wid = int(width/6)
        if multi_wid*6 != width:
            print("hihi")
            combo.SetSelection(4)
        combo.SetSelection(3)

       
    def listBoxFocus(self, event):
        print("ListBox receives focus")
        self.changeTool("ListBox")
        event.Skip()

    def textCtrlFocus(self, event):
        print("TextCtrl receives focus")
        self.changeTool("TextCtrl")
        event.Skip()

    def comboBoxFocus(self, event):
        print("ComboBox receives focus")
        self.changeTool("ComboBox")
        event.Skip()

    def gaugeClick(self, event):
        print("Gauge clicked...")
        self.changeTool("Gauge")
        event.Skip()

    def sliderFocus(self, event):
        print("Slider receives focus")
        self.changeTool("Slider")
        event.Skip()

    def spinCtrlFocus(self, event):
        print("SpinCtrl receives focus")
        self.changeTool("SpinCtrl")
        event.Skip()

    def changeTool(self, name):
        connectMessage = {
            "message_type": "tool_change",
            "session_id": sessionId,
            "tool_id": name
        }
        regMsg =  json.dumps(connectMessage)
        ws.send(regMsg.encode('utf8'))


if __name__ == '__main__':
    global ws
    global craft

    app = wx.App()
    frame = TestFrame(parent=None, id=-1)
    frame.Show()

    craft = CraftClient()

    if platform.system() == 'Windows':
        craft.connect("Craft.exe", "")
    else:
        craft.connect("craft.app", "")

app.MainLoop()
