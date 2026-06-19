#Requires AutoHotkey v2.0
#SingleInstance Off

; GUI automation helper for tools/typora_pdf_export.py.
; Python owns scanning, retry, logging, resume, and PDF checks.
; AutoHotkey owns Typora GUI actions only.

SetTitleMatchMode 2
SendMode "Input"
DetectHiddenWindows false

Log(message) {
    FileAppend("GUI " message "`n", "*")
}

Fail(message, code := 1) {
    Log("ERROR " message)
    ExitApp(code)
}

RequireArg(index, name) {
    if (A_Args.Length < index || A_Args[index] = "") {
        Fail("Missing argument: " name, 10)
    }
    return A_Args[index]
}

WindowText(hwnd) {
    title := ""
    text := ""
    try title := WinGetTitle("ahk_id " hwnd)
    try text := WinGetText("ahk_id " hwnd)
    return title "`n" text
}

WindowContains(hwnd, patterns) {
    haystack := WindowText(hwnd)
    for pattern in patterns {
        if InStr(haystack, pattern) {
            return true
        }
    }
    return false
}

ClickButtonByText(hwnd, labels) {
    controls := []
    try controls := WinGetControls("ahk_id " hwnd)
    for controlName in controls {
        controlText := ""
        try controlText := ControlGetText(controlName, "ahk_id " hwnd)
        if (controlText = "") {
            continue
        }
        for label in labels {
            if InStr(controlText, label) {
                try {
                    ControlClick(controlName, "ahk_id " hwnd)
                    return true
                }
            }
        }
    }
    return false
}

ActivationPatterns() {
    return ["激活 Typora", "您的试用还有", "Activate Typora", "trial", "Trial"]
}

LaterLabels() {
    return ["以后再说", "Later", "later"]
}

FindActivationPopup() {
    patterns := ActivationPatterns()
    for hwnd in WinGetList() {
        if WindowContains(hwnd, patterns) {
            return hwnd
        }
    }
    return 0
}

PopupStillVisible(hwnd) {
    return WinExist("ahk_id " hwnd) && WindowContains(hwnd, ActivationPatterns())
}

TryTabToLaterButton(hwnd) {
    labels := LaterLabels()
    Loop 20 {
        Send("{Tab}")
        Sleep(120)
        focused := ""
        focusedText := ""
        try focused := ControlGetFocus("ahk_id " hwnd)
        if (focused != "") {
            try focusedText := ControlGetText(focused, "ahk_id " hwnd)
        }
        for label in labels {
            if InStr(focusedText, label) {
                Send("{Enter}")
                Log("clicked later button")
                return true
            }
        }
    }

    ; Electron windows may not expose focused button text. Use a bounded Tab
    ; sequence plus Enter, then the caller verifies the popup disappeared.
    Send("{Tab 8}")
    Sleep(120)
    Send("{Enter}")
    Log("clicked later button")
    return true
}

DismissActivationPopup(hwnd) {
    title := ""
    try title := WinGetTitle("ahk_id " hwnd)
    Log("activation popup detected: " title)

    WinActivate("ahk_id " hwnd)
    if !WinWaitActive("ahk_id " hwnd, , 3) {
        Log("activation popup could not be activated immediately")
    }
    Sleep(250)

    if ClickButtonByText(hwnd, LaterLabels()) {
        Log("clicked later button")
    } else {
        TryTabToLaterButton(hwnd)
    }

    Sleep(700)
    if !PopupStillVisible(hwnd) {
        Log("popup dismissed")
        return true
    }

    Send("{Esc}")
    Sleep(500)
    if !PopupStillVisible(hwnd) {
        Log("popup dismissed")
        return true
    }

    Log("popup still present after fallback")
    return false
}

DismissActivationPopups(maxWaitMs := 15000) {
    deadline := A_TickCount + maxWaitMs
    sawPopup := false

    while (A_TickCount < deadline) {
        hwnd := FindActivationPopup()
        if (!hwnd) {
            if !sawPopup {
                Log("popup check complete: none detected")
            }
            return sawPopup
        }

        sawPopup := true
        DismissActivationPopup(hwnd)
        Sleep(500)
    }

    hwnd := FindActivationPopup()
    if (hwnd) {
        Log("popup still present after fallback")
        Fail("Typora activation popup still present after 15 seconds", 60)
    }
    Log("popup dismissed")
    return sawPopup
}

ActivateTypora(typoraExe, timeoutSeconds := 20) {
    if !WinExist("ahk_exe Typora.exe") {
        Log("Typora not running; launching")
        Run('"' typoraExe '"')
    }
    if !WinWait("ahk_exe Typora.exe", , timeoutSeconds) {
        Fail("Timed out waiting for Typora window", 20)
    }
    DismissActivationPopups(15000)
    WinActivate("ahk_exe Typora.exe")
    if !WinWaitActive("ahk_exe Typora.exe", , timeoutSeconds) {
        DismissActivationPopups(15000)
        WinActivate("ahk_exe Typora.exe")
        if !WinWaitActive("ahk_exe Typora.exe", , timeoutSeconds) {
            Fail("Timed out activating Typora window", 21)
        }
    }
    DismissActivationPopups(15000)
    Log("Typora activated")
}

WaitActiveDialog(timeoutSeconds, label) {
    deadline := A_TickCount + timeoutSeconds * 1000
    while (A_TickCount < deadline) {
        DismissActivationPopups(15000)
        if WinActive("ahk_class #32770") {
            title := ""
            try title := WinGetTitle("A")
            Log(label " dialog detected: " title)
            return true
        }
        Sleep(250)
    }
    Fail("Timed out waiting for " label " dialog", 30)
}

WaitSaveDialog(timeoutSeconds) {
    deadline := A_TickCount + timeoutSeconds * 1000
    savePatterns := ["Save", "Save As", "另存", "保存", "Export", "导出"]
    while (A_TickCount < deadline) {
        DismissActivationPopups(15000)
        active := WinExist("A")
        if active && WinActive("ahk_class #32770") && WindowContains(active, savePatterns) {
            title := ""
            try title := WinGetTitle("ahk_id " active)
            Log("save dialog detected: " title)
            return true
        }
        if active && WinActive("ahk_class #32770") {
            title := ""
            try title := WinGetTitle("ahk_id " active)
            Log("save dialog detected: " title)
            return true
        }
        Sleep(250)
    }
    Fail("Timed out waiting for Save dialog", 31)
}

PasteText(text) {
    oldClipboard := ClipboardAll()
    A_Clipboard := text
    if !ClipWait(3) {
        Fail("Timed out setting clipboard", 40)
    }
    Send("^v")
    Sleep(200)
    A_Clipboard := oldClipboard
}

WaitForDocumentLoad(markdownPath, loadWaitMs) {
    SplitPath(markdownPath, &fileName)
    deadline := A_TickCount + loadWaitMs
    seenTitle := false
    stableStart := 0
    lastTitle := ""

    while (A_TickCount < deadline) {
        DismissActivationPopups(15000)
        if WinActive("ahk_exe Typora.exe") {
            title := ""
            try title := WinGetTitle("A")
            if InStr(title, fileName) {
                seenTitle := true
            }
            if (title != "" && title = lastTitle) {
                if (stableStart = 0) {
                    stableStart := A_TickCount
                }
                if seenTitle && (A_TickCount - stableStart > 700) {
                    Log("document load wait complete: " title)
                    return true
                }
            } else {
                lastTitle := title
                stableStart := 0
            }
        }
        Sleep(250)
    }
    Log("document load wait timeout reached; continuing after fixed wait")
    return true
}

OpenMarkdownInTypora(typoraExe, markdownPath, loadWaitMs) {
    ActivateTypora(typoraExe, 30)
    DismissActivationPopups(15000)
    Send("^o")
    WaitActiveDialog(20, "open")
    Sleep(250)
    Send("^a")
    PasteText(markdownPath)
    Send("{Enter}")
    if !WinWaitActive("ahk_exe Typora.exe", , 30) {
        DismissActivationPopups(15000)
        if !WinWaitActive("ahk_exe Typora.exe", , 10) {
            Fail("Timed out returning to Typora after opening file", 50)
        }
    }
    ActivateTypora(typoraExe, 10)
    WaitForDocumentLoad(markdownPath, loadWaitMs)
}

ConfirmOverwriteIfPresent(waitSeconds := 5) {
    confirmPatterns := ["Confirm Save As", "Confirm", "Replace", "Overwrite", "already exists", "确认", "替换", "覆盖", "已存在"]
    yesLabels := ["Yes", "&Yes", "是", "确定", "替换", "保存"]
    deadline := A_TickCount + waitSeconds * 1000
    while (A_TickCount < deadline) {
        active := WinExist("A")
        if active && WinActive("ahk_class #32770") && WindowContains(active, confirmPatterns) {
            title := ""
            try title := WinGetTitle("ahk_id " active)
            if ClickButtonByText(active, yesLabels) {
                Log("overwrite confirmed by button text: " title)
            } else {
                Send("!y")
                Sleep(250)
                if WinActive("ahk_class #32770") {
                    Send("{Enter}")
                }
                Log("overwrite confirmed by keyboard: " title)
            }
            return true
        }
        Sleep(250)
    }
    Log("overwrite check complete: no confirmation detected")
    return false
}

ExportWithPreviousSettings(typoraExe, dialogTimeoutSeconds) {
    ActivateTypora(typoraExe, 10)
    DismissActivationPopups(15000)
    Log("export shortcut sent")
    Send("^+e")
    WaitSaveDialog(dialogTimeoutSeconds)
    Sleep(250)
    Send("{Enter}")
    Log("save dialog accepted with Enter")
    ConfirmOverwriteIfPresent(5)
}

CloseCurrentDocument(typoraExe, closeWaitMs := 500) {
    ActivateTypora(typoraExe, 20)
    Send("^w")
    Sleep(closeWaitMs)

    if WinActive("ahk_class #32770") {
        Send("{Esc}")
        Sleep(200)
        Log("close prompt dismissed with Escape")
    }
}

action := RequireArg(1, "action")

if (action = "open_export") {
    typoraExe := RequireArg(2, "typora_exe")
    markdownPath := RequireArg(3, "markdown_path")
    loadWaitMs := Integer(RequireArg(4, "load_wait_ms"))
    dialogTimeoutSeconds := Integer(RequireArg(5, "dialog_timeout_seconds"))
    OpenMarkdownInTypora(typoraExe, markdownPath, loadWaitMs)
    ExportWithPreviousSettings(typoraExe, dialogTimeoutSeconds)
    ExitApp(0)
} else if (action = "close") {
    typoraExe := RequireArg(2, "typora_exe")
    closeWaitMs := 500
    if (A_Args.Length >= 3 && A_Args[3] != "") {
        closeWaitMs := Integer(A_Args[3])
    }
    CloseCurrentDocument(typoraExe, closeWaitMs)
    ExitApp(0)
} else if (action = "activate") {
    typoraExe := RequireArg(2, "typora_exe")
    ActivateTypora(typoraExe, 30)
    ExitApp(0)
} else {
    Fail("Unknown action: " action, 11)
}
