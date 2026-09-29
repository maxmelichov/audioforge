"""v5 shot list: page, data, segments (see film.py), fine print (short variant for 1:1). Every number on screen is read
by the page from data.json (demo/numbers.json + runs/*.json via demo/v5/export.py, "plain" section for the result
charts); the fine print names the source. The real examples come from demo/v5/examples.py (examples.json)."""

E2E = "runs/e2e_final.json"
AGENT = "agent voice added for the demo at the recorded cut-in and response times"
SUBS = "subtitles: the speaker's words (reference transcript)"

SHOTS = {
    "coldopen": dict(breathe=True, page="coldopen.html", clip="ex_call_today", still=9.0,
                     segments=[("say", "call_1"), ("clip", "P", None, None, ""), ("say", "call_2")],
                     fine="A real two-person call (TurnBench) through Pipecat 1.12.0\'s default stack, live · agent voice added at the recorded time",
                     fine_sq="Real call, Pipecat 1.12.0 default, live · agent voice added"),
    "promise": dict(page="promise.html", still=3.0, min_dur=3.6, breathe=True, segments=[("say", "promise")], fine="", fine_sq=""),
    "arch": dict(page="arch2.html", clip="room_IS1008b_1670s", still=24.0,
                 segments=[("say", "arch_1"), ("say", "arch_2"), ("say", "arch_4"), ("say", "arch_5"), ("hold", 1.5)],
                 fine="Speaker error: within-meeting EER on AMI, all-block mix vs block-4 head · compute: same outputs, 0 decision changes",
                 fine_sq="Within-meeting speaker EER, AMI · same outputs, 0 decision changes"),
    "setup_pc": dict(page="setup.html", still=3.0, min_dur=3.6, tail=0.2, breathe=True, segments=[("say", "setup_pc")],
                     params={"title": "Same Pipecat pipeline, same voice", "rows": [["Pipecat 1.12.0", "Silero VAD · smart-turn v3.2 · default settings"], ["words", "Whisper small (Pipecat default: distil-medium.en)"]],
                             "swap": "The only difference is the listening."},
                     fine="Silero VAD 0.7 confidence, 0.2 s start and stop · smart-turn v3.2 with its 3 s fallback",
                     fine_sq="Silero defaults · smart-turn v3.2, 3 s fallback"),
    "setup_lk": dict(page="setup.html", still=3.0, min_dur=3.6, tail=0.2, breathe=True, segments=[("say", "words_1")],
                     params={"title": "LiveKit Agents, default settings", "rows": [["LiveKit 1.8.3", "Silero VAD · English turn detector · defaults"], ["words", "Whisper small (LiveKit ships no local STT)"]],
                             "swap": "The only difference is the listening."},
                     fine="Endpointing 0.5 / 3.0 s · preemptive generation off",
                     fine_sq="Endpointing 0.5 / 3.0 s · preemptive generation off"),
    "ex_interrupt": dict(light=2.2, breathe=True, page="split.html", clip="ex_interrupt", panels={"L": "L", "R": "R"}, mode="cutins", notext=True, subs=True, still=20.0,
                         segments=[("clip", "LR", None, None, ""), ("say", "ex_int_3")],
                         fine="16 real calls: Pipecat\'s default cuts in 2.2× a minute, audioforge 1.3×; this is a typical stretch · agent voice added at the recorded times",
                         fine_sq="16 real calls: 2.2 vs 1.3 cut-ins a minute; a typical stretch"),
    "ex_words": dict(noagent=True, light=2.2, breathe=True, page="split.html", clip="ex_words", panels={"L": "L", "R": "R"}, mode="words", subs=True, still=10.5,
                     segments=[("clip", "LR", None, None, ""), ("say", "words_2")],
                     fine="Medians on real calls: 3.4 s vs 1.2 s · left text: reference transcript at the measured first-word time · right text: a live run of the same audio",
                     fine_sq="Medians on real calls: 3.4 s vs 1.2 s · left text: reference at the measured time"),
    "ex_phone": dict(light=2.2, breathe=True, page="split.html", clip="ex_phone", panels={"L": "L", "R": "R"}, mode="cutins", notext=True, subs=True, still=8.8,
                     segments=[("say", "phone_1"), ("clip", "LR", None, None, ("Left: Pipecat default · Right: audioforge", "Left: default · Right: audioforge")), ("say", "phone_2")],
                     fine=f"otoSpeech real phone call (CC-BY-4.0), the caller's own channel, live through Pipecat 1.12.0 at 1x · {AGENT} · subtitles: automatic transcript (no reference exists) · {E2E}",
                     fine_sq=f"otoSpeech call (CC-BY-4.0), Pipecat 1.12.0, live · {AGENT}"),
    "ex_room": dict(light=2.2, breathe=True, page="room_example.html", clip="ex_room", subs=True, still=8.0,
                    segments=[("say", "room_1"), ("clip", "R", None, None, ""), ("say", "room_2")],
                    fine="AMI meeting IS1008b, overlapping speech · speaker labels: live diarizer · subtitles: reference transcript",
                    fine_sq="AMI IS1008b · live diarizer · reference subtitles"),
    # result charts: the CHARTS agent's pages (numbers_b.js, their own fine print); narration lines in script.md
    "live_answers": dict(skip=0.9, page="chart_live.html", params={"frame": "answers"}, still=5.0, min_dur=6.0, segments=[("say", "live_answers")]),
    "live_interrupts": dict(skip=0.9, page="chart_live.html", params={"frame": "interrupts"}, still=3.6, min_dur=4.6, segments=[("say", "live_interrupts")]),
    "live_first": dict(skip=0.9, page="chart_live.html", params={"frame": "first"}, still=4.2, min_dur=5.2, segments=[("say", "live_first")]),
    "chart_icsi": dict(skip=0.9, page="chart_icsi.html", still=4.6, min_dur=5.6, segments=[("say", "chart_icsi")]),
    "chart_turnbench": dict(skip=0.9, page="chart_turnbench.html", still=4.6, min_dur=5.6, segments=[("say", "chart_turnbench")]),
    "chart_wer": dict(skip=0.9, page="chart_wer.html", still=4.6, min_dur=5.6, segments=[("say", "chart_wer")]),
    "finale": dict(page="finale.html", params={"wer": True}, still=12.5, min_dur=14.4, segments=[("say", "finale")]),
    "end": dict(page="end.html", still=3.0, min_dur=5.4, tail=1.8, segments=[("say", "end")]),
}

CUTS = {
    "v6": ["coldopen", "promise", "arch", "setup_pc", "ex_interrupt", "setup_lk", "ex_words", "ex_room",
             "finale", "end"],
}
