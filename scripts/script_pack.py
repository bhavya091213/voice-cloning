"""The recording scripts of the Studio's Voices tab: the single source for the recorder UI and for enroll_multi.py.

A section is one recording (parts/<id>.wav). Its items are read in order; an item is (tag, direction, text).
  tag        names the intonation clip cut from that item (it becomes @tag in scripts and a clip in the .vcref); consecutive
             items with the same tag are joined into one clip, so give a tag to 2-3 short lines to reach 4-15 s.
             None = recorded for cadence/coverage only.
  direction  how to deliver it (shown to the speaker, never spoken).
A, B and C are the original three parts; enroll_multi.py still handles them with its own logic (legacy=True).
"""
from collections import namedtuple

Item = namedtuple("Item", "tag direction text")
Section = namedtuple("Section", "id group title minutes goal how items legacy")


def S(id, group, title, minutes, goal, how, items, legacy=False):
    return Section(id, group, title, minutes, goal, how, [Item(*i) for i in items], legacy)


SECTIONS = [
    S("A", "Core", "Neutral narration", 1.0, "Baseline voice, timbre and everyday pace.",
      "Warm, steady, conversational. Normal pace, like reading to one person.", [
          (None, "", "The big brown fox jumped quickly over the fence, then paused beside a quiet stream."),
          (None, "", "Thirty-three thieves thought they thrilled the throne throughout Thursday."),
          (None, "", "She sells seashells by the shore, and the measure of pleasure is a rare treasure."),
          (None, "", "Judge George chose a large orange from the church's garden."),
          (None, "", "\"Why would you ask?\" Yvonne said. \"We all know the answer, don't we?\""),
          (None, "", "A cheerful young man could hardly wait to bring home a loaf of fresh bread, a few green beans, and some very ripe peaches."),
          (None, "", "The pale moon rose over the vast, calm lake; a gentle breeze moved through the tall pines, and far off, a lone bird called."),
          (None, "", "Please call Paul at five-oh-nine, eight-seven-six, four-three-two-one, or send it by noon on June twenty-ninth."),
          (None, "", "Dad took the book out of the rug-covered box; Ken's boat rode the foam toward home."),
      ], legacy=True),
    S("B", "Core", "Six core emotions", 0.7, "The first six emotional colours.",
      "Do each line fully in the stated emotion. Commit to it; this drives emotional range.", [
          ("excited", "Excited, fast", "No way! You actually got it done? That is absolutely incredible, I knew you could do it!"),
          ("warm", "Warm, soft, slow", "Take your time. There's no rush at all. Everything is going to be just fine."),
          ("serious", "Serious, low, deliberate", "Listen carefully. What happens next matters more than anything we've done so far."),
          ("sad", "Sad, quiet", "I really thought it would be different this time. I miss how things used to be."),
          ("annoyed", "Annoyed, sharp", "Seriously? Again? I told you twice already. Just put it back where it belongs."),
          ("playful", "Playful, teasing", "Oh, you think you're so clever, don't you? Well, we'll just see about that."),
      ], legacy=True),
    S("C", "Core", "Short utterances", 0.3, "Reactions and one-word answers.",
      "Pause about a second between each. Say them like real replies, not like a list.", [
          (None, "", "Yes. No. Maybe. Right. Okay, sure. Hmm, let me think. Really? What do you mean? Wait, hold on. Thank you so much. Ha, that's great. Ahh, finally."),
      ], legacy=True),

    S("D", "Emotion", "Wider emotional range", 3.5, "Ten more emotions plus voice-quality extremes; each line is its own clip.",
      "Two lines per emotion. Act it, do not describe it. Big enough that a stranger would hear the feeling with eyes closed.", [
          ("angry", "Angry, hard, loud", "I said no. I am done asking nicely. You knew exactly what you were doing, and you did it anyway."),
          ("angry", "Angry, hard, loud", "Get out. Get out right now before I say something I can't take back."),
          ("nervous", "Nervous, tight, a little fast", "Okay, so, um, I need to tell you something, and please don't be mad."),
          ("nervous", "Nervous, tight, a little fast", "I think I might have, well, maybe broken it. Just a little. Probably."),
          ("whisper", "Whispered, close to the mic, intimate", "Keep your voice down. They're right outside. Don't move, and don't say a word."),
          ("whisper", "Whispered", "Come closer. I'll tell you the secret, but you have to promise never to repeat it."),
          ("confident", "Confident, chest-forward, easy smile", "I've done this a hundred times. Trust me, it's going to work, and it's going to be better than you expect."),
          ("tender", "Tender, soft, slow, loving", "Hey. Hey, look at me. You did so well today, and I am so proud of you."),
          ("tender", "Tender", "Sleep now. I'll be right here. Nothing is going to hurt you."),
          ("sarcastic", "Sarcastic, dry, exaggerated", "Oh, great. Another meeting that could have been an email. My favourite."),
          ("sarcastic", "Sarcastic", "Wow, what a surprise. The one thing that could go wrong went wrong. Who could have possibly seen that coming?"),
          ("disappointed", "Disappointed, deflated, falling", "I just expected more from you. I really did. That's what hurts the most."),
          ("surprised", "Surprised, sudden, high", "Wait, what? You're here? Oh my gosh, I had no idea you were coming!"),
          ("amused", "Amused, laughing through the words", "Okay, okay, stop, you can't say that with a straight face. I can't. That is hilarious."),
          ("tired", "Tired, heavy, slow, low energy", "It's been such a long day. I just want to sit down and not think about anything for a while."),
          ("urgent", "Urgent, fast, pressing", "We have to go now. Grab your things, leave the rest, and don't stop until we're at the car."),
          ("calm", "Calm, even, soothing, unhurried", "Breathe in slowly. Hold it for a moment. And let it go. There is nothing you need to do right now."),
          ("proud", "Proud, warm, lifted", "Look at what we built. Every late night, every wrong turn, all of it was worth it."),
          ("fearful", "Fearful, shaky, breathy", "Did you hear that? Something is moving in the dark. Please tell me you heard that too."),
          ("deadpan", "Deadpan, flat, no emotion at all", "I am thrilled. Truly. This is the happiest moment of my entire life."),
          ("awe", "Awe, hushed, wondering", "Oh wow. Look at that. I've never seen anything so beautiful in my whole life."),
          ("shout", "Shouting across a distance, projected", "Hey! Over here! Can you hear me? We're over here, by the old bridge!"),
          ("gentle_laugh", "Laugh softly while speaking", "Ha, I know, I know. It sounds ridiculous when you say it out loud, doesn't it?"),
      ]),

    S("E", "Questions", "Questions and rising/falling intonation", 3.0, "Every common question shape, because pitch contour is what questions live on.",
      "Let the pitch do the work: yes/no questions rise at the end, wh-questions fall, lists rise-rise-fall. Do not over-act.", [
          ("q_yesno", "Rising at the end", "Are you coming with us tonight? Did you remember to lock the door? Is that really what she said?"),
          ("q_wh", "Falling at the end", "Where did you put the keys? What time does the meeting start? Why would anyone do that? How long have you been waiting?"),
          ("q_tag", "Tag question, seeking agreement", "It's a lovely day, isn't it? You've been here before, haven't you? We should probably go, shouldn't we?"),
          ("q_rhetorical", "Rhetorical, weary, no answer expected", "What did you expect would happen? Who does that? Is it really so hard to just say sorry?"),
          ("q_incredulous", "Incredulous, climbing sharply", "You did what? He said that to your face? You're telling me it was free the whole time?"),
          ("q_polite", "Polite and soft", "Would you mind if I opened the window? Could I ask you a quick favour? Do you happen to know where the station is?"),
          ("q_choice", "Choice question: rise on the first option, fall on the last", "Would you like coffee, tea, or water? Are we going left or right? Do you want it today or tomorrow?"),
          ("q_followup", "Curious, leaning in", "And then what happened? Wait, so who told you? Okay, but how did you find out?"),
          ("q_clarify", "Clarifying, careful", "Sorry, just to be sure, do you mean this Friday or next Friday? So the deadline is the fifteenth, right?"),
          ("q_hesitant", "Hesitant, trailing off", "Um, do you think maybe we could, I don't know, try it again? Is that okay? Or... never mind?"),
          ("q_interview", "Warm interviewer", "Tell me about the moment you knew this was what you wanted to do. What was going through your mind?"),
          ("q_quiz", "Bright quiz host", "Here's your next question. Which planet is closest to the sun? Take your time. Is it A, B, C, or D?"),
          (None, "Answer calmly, as a statement", "That's a good question. The honest answer is that nobody knows yet."),
      ]),

    S("F", "Conversation", "Natural conversation", 3.0, "Fillers, reactions, interruptions and the loose rhythm of real talk.",
      "Talk, do not read. Use the little hesitations, restarts and 'um's as written. Smile where it fits.", [
          ("chat_casual", "Relaxed, chatty, fast and loose", "So, yeah, I was gonna call you earlier, but then my phone died, and, honestly, the whole day just kind of got away from me."),
          ("chat_casual", "Relaxed", "Anyway, long story short, we ended up getting pizza, which, you know, is never a bad outcome."),
          ("chat_react", "Genuine reactions, short and varied", "Oh, wow. No way. Seriously? Huh. That's wild. I did not see that coming."),
          ("chat_react", "Backchannel", "Mm-hm. Right, right. Yeah, yeah, totally. Oh, okay, that makes sense."),
          ("chat_story", "Telling an anecdote, building up", "So, there I am, standing in the middle of the airport, no passport, no phone, and the gate is closing in ten minutes."),
          ("chat_story", "Punchline, quick then a beat", "And the guy just looks at me, completely serious, and says, sir, this is a bus station."),
          ("chat_correction", "Self-correcting mid-sentence", "It was on Tuesday, no, wait, Wednesday. Sorry, it was definitely Wednesday, because the market was closed."),
          ("chat_aside", "Lower, quicker, said under your breath", "He said he'd be on time, which, I mean, we all know how that goes, but anyway, we waited."),
          ("chat_agree", "Warm agreement", "Yes, exactly, that's what I was trying to say. You put it so much better than I did."),
          ("chat_disagree", "Polite pushback", "I hear you, but I'm not sure that's quite right. Can I offer a different way of looking at it?"),
          ("chat_thinking", "Thinking aloud, slow with pauses", "Hmm. Let me think. Okay, so if we move that to Monday, then, no, that won't work. Unless, hold on."),
          ("chat_laugh", "Laughing, then recovering", "Ha ha, oh no. Sorry. Okay. I'm fine. I'm fine. Where was I?"),
          ("chat_greet", "Bright greetings and goodbyes", "Hey, good to see you! How've you been? Come on in. Anyway, I've got to run, so talk soon, take care, bye!"),
          ("chat_empathy", "Sympathetic, low and slow", "Oh, I'm so sorry. That sounds really hard. I'm here if you want to talk about it."),
      ]),

    S("G", "Explaining", "Explainer and lecture voice", 5.0,
      "Teaching cadence: how clauses run on, where emphasis lands, how a recap sounds. This is the register most scripts use.",
      "A clear, friendly teacher. Steady pace, slightly lifted energy on key terms. Commas are short breaths, full stops are real pauses.", [
          ("exp_intro", "Opening a topic, welcoming", "In this lesson, we'll look at how memory works in the brain. By the end, you'll know the three stages, and why forgetting is not always a bad thing."),
          ("exp_define", "Defining a term, precise", "Working memory is the small amount of information you hold in mind while you use it. Think of it as a mental notepad that is constantly being erased."),
          ("exp_list", "A run of clauses, each one a step", "To build a good habit, start with something tiny, attach it to something you already do, repeat it every day, and celebrate when you finish."),
          ("exp_list", "Same rhythm, new list", "A strong summary names the main idea, drops the small details, keeps the order of events, and uses your own words."),
          ("exp_contrast", "Setting two things against each other", "Short-term memory holds a few things for a few seconds, whereas long-term memory can hold a lifetime of experience."),
          ("exp_contrast", "Contrast with a turn", "It might look like a shortcut, but it's actually the long way around. The slow route is faster in the end."),
          ("exp_example", "Making it concrete", "For example, imagine you're trying to remember a phone number. You repeat it, you break it into chunks, and you picture where you'll write it down."),
          ("exp_example", "Second example, friendlier", "Say you're learning a new language. At first, every word is a puzzle. A month later, whole sentences simply arrive."),
          ("exp_warn", "Careful, slightly slower", "Be careful here, because this is where most people go wrong. Never assume the first answer is the correct one. Always check it."),
          ("exp_emphasis", "Stress the starred words", "What matters most is not how *fast* you work, but how *well* you work. Quality first. Speed will follow."),
          ("exp_question", "Posing a question to the listener", "So why does this matter? Because every decision you make depends on it. And what happens when it's wrong? That's what we'll explore next."),
          ("exp_transition", "Moving to the next idea", "That covers the basics. Now, let's turn to something a little more advanced, and see how these ideas fit together."),
          ("exp_recap", "Recapping, settled and conclusive", "To recap, we covered three things: how memory is stored, how it is recalled, and how to make it last. Together, they form the foundation for everything that follows."),
          ("exp_close", "Closing, warm and final", "Thanks for watching. If this was helpful, try the practice exercise below, and I'll see you in the next lesson."),
      ]),

    S("H", "Storytelling", "Storytelling and audiobook", 4.0, "Narrative pacing: description, action, suspense and dialogue between characters.",
      "Narrator voice, a little theatrical. Give each character a different colour, but stay yourself.", [
          ("story_scene", "Scene-setting, unhurried, descriptive", "The village lay quiet beneath a grey and heavy sky. Smoke curled from a dozen chimneys, and somewhere beyond the hill, a bell rang slowly, once, twice, and then no more."),
          ("story_action", "Action, quick and breathless", "She ran. Down the stairs, across the courtyard, through the gate, and into the rain, with the footsteps closing in behind her."),
          ("story_suspense", "Suspense, quiet and slow, long pauses", "The door was open. It had not been open this morning. He stood very still, and listened. Nothing. And then, from upstairs, a single creak."),
          ("story_char_a", "Character one: older, gravelly, weary", "\"I've seen this storm before,\" the old sailor said. \"It never ends well for the ones who stay.\""),
          ("story_char_b", "Character two: young, bright, impatient", "\"Then we'd better go,\" the girl said, already pulling on her coat. \"Come on! We're wasting time!\""),
          ("story_wonder", "Wonder, rising", "And there, at the edge of the world, the mountains turned to gold, as though the sun itself had decided to stay a little longer."),
          ("story_humor", "Wry, light, a smile in the voice", "It was, by every measure, a terrible plan. Naturally, they were delighted with it."),
          ("story_reflect", "Reflective, inward, slow", "Years later, she would remember that evening differently. Not the words, but the quiet between them, and the way the light had fallen across the table."),
          ("story_end", "Ending, soft landing", "And so they walked home together, tired and happy, as the first stars came out. The end."),
      ]),

    S("I", "Performance", "Ads, news, podcast and trailer reads", 3.5, "Broadcast registers with their own pace, punch and energy.",
      "Pro announcer mode. Smile for ads, steady for news, intimate for podcast, huge for trailer.", [
          ("ad_hype", "High-energy retail ad", "Big savings are here! For one weekend only, everything is half price. Don't wait, don't wonder, and don't miss out!"),
          ("ad_warm", "Warm, trustworthy brand film", "Some things are made to last. Built by hand, carried through generations, and always there when you need them."),
          ("ad_cta", "Call to action, clear and friendly", "Visit our website today, sign up in under a minute, and get your first month free. That's it. Simple as that."),
          ("news_read", "Neutral news anchor, even and crisp", "Officials announced on Tuesday that the new transport line will open in spring, following two years of delays and a revised budget of four billion dollars."),
          ("news_sport", "Sports commentary, rising excitement", "He takes it down the left wing, cuts inside, shoots, and scores! What a goal! The crowd is on its feet!"),
          ("trailer", "Movie-trailer voice, low and grand", "In a world on the edge of darkness, one choice will change everything. This summer, the fight begins."),
          ("podcast_intro", "Podcast host, relaxed, close mic", "Hey everyone, welcome back to the show. I'm so glad you're here. Today we have a really fun one, so grab a coffee and settle in."),
          ("podcast_outro", "Podcast sign-off, grateful", "That's all for today. Thank you so much for listening. If you liked it, tell a friend, and I'll catch you next week."),
          ("announce", "Public announcement, slow and clear", "Attention please. The next train to the airport will depart from platform four in five minutes. Please stand clear of the doors."),
          ("meditation", "Guided meditation, very slow and low", "Let your shoulders drop. Let your jaw soften. With every breath out, let a little more of the day fall away."),
          ("product_demo", "Friendly tutorial voice", "First, open the settings menu. Then tap the profile icon, choose your preferences, and press save. That's all there is to it."),
      ]),

    S("J", "Pacing", "Speed and pause control", 3.0, "Same words at three speeds, plus planned pauses. This is how the model learns what slow and fast really mean for you.",
      "Follow the pace in the direction exactly. Leave the pauses where they are written (a '...' is about a second of silence).", [
          ("pace_slow", "SLOW, almost deliberate, stretch the vowels", "The meeting has been moved to Thursday afternoon, so please update your calendar and let your team know."),
          ("pace_slow", "SLOW", "Take one step, then another, and keep your eyes on the horizon until you reach the top."),
          ("pace_normal", "NORMAL, everyday conversational speed (same two sentences again)", "The meeting has been moved to Thursday afternoon, so please update your calendar and let your team know."),
          ("pace_normal", "NORMAL", "Take one step, then another, and keep your eyes on the horizon until you reach the top."),
          ("pace_fast", "FAST, brisk but still clear, no slurring (same two sentences again)", "The meeting has been moved to Thursday afternoon, so please update your calendar and let your team know."),
          ("pace_fast", "FAST", "Take one step, then another, and keep your eyes on the horizon until you reach the top."),
          ("pause_dramatic", "Dramatic pauses at the dots", "And the winner is... ... ... you. Not because you were the fastest... but because you never stopped."),
          ("pause_comma", "Clear comma breaths, short and even", "First, we mix. Then, we wait. After that, we fold, we rest, we bake, and we cool."),
          ("pause_para", "Long pause between each sentence (about a second)", "That was the first part. ... Now for the second part. ... And finally, the last part."),
          ("pace_accel", "Start slow, speed up through the sentence, end fast", "It began as a whisper, then a murmur, then a rumble, and before anyone could react, the whole room was roaring."),
          ("pace_decel", "Start brisk, slow down through the sentence, end very slow", "Everything was moving so fast, until it all began to slow, and settle, and finally fall quiet."),
      ]),

    S("K", "Pacing", "Emphasis and stress", 1.5, "Same sentence, different word stressed. Pitch and duration on one word change the meaning.",
      "Lean hard on the CAPITAL word only. Everything else stays flat.", [
          ("emph_a", "Stress the first word", "I didn't say she stole it. ... I DIDN'T say she stole it."),
          ("emph_a", "Stress 'say'", "I didn't SAY she stole it. ... I didn't say SHE stole it."),
          ("emph_b", "Stress 'stole'", "I didn't say she STOLE it. ... I didn't say she stole IT."),
          ("emph_c", "Contrast pair", "I wanted the BLUE one, not the green one. ... I wanted the blue ONE, not two."),
          ("emph_d", "Emphasis on numbers and names", "It costs fifteen dollars, not FIFTY. ... Her name is Dana, not DANIEL. ... It's on page NINE."),
      ]),

    S("L", "Coverage", "Numbers, names, terms and symbols", 2.5, "Things TTS gets wrong. Real examples of how you say them aloud.",
      "Say them the way you would to a friend. Do not spell out punctuation.", [
          ("tech_ai", "Technical but natural", "Large language models predict the next token. The GPU runs the inference, the API returns the response, and the latency depends on the context length."),
          ("tech_ai", "Technical but natural", "AI fluency means working with AI effectively, efficiently, ethically, and safely."),
          ("tech_numbers", "Clear numbers", "The total came to two thousand, four hundred and fifty dollars, which is up three point five percent from last year."),
          ("tech_numbers", "Clear numbers", "Call 555-0182, extension 4, between nine a.m. and five-thirty p.m., Monday to Friday."),
          ("tech_dates", "Dates and times", "The event is on the twenty-first of March, twenty twenty-six, at seven-fifteen in the evening, in room B-204."),
          ("tech_acronyms", "Acronyms said as letters or words, the natural way", "The CEO sent an FAQ to the HR team about the NASA visit, the USB drives, and the new PDF forms."),
          ("tech_names", "Names and places, confident", "Dr. Nguyen flew from Reykjavik to Buenos Aires, then on to Marrakech, to meet Siobhan and Aleksandr."),
          ("tech_units", "Units", "It weighs about two and a half kilograms, runs at three point two gigahertz, and uses twelve watts."),
          ("tech_symbols", "Read the symbols naturally", "Save twenty percent, buy one get one free, and email us at support at example dot com."),
          ("tech_quote", "Quotation, then comment", "She said, \"Never trust a first draft,\" and honestly, she was right."),
      ]),

    S("M", "Coverage", "Sounds and tongue-twisters", 2.0, "Phonetic coverage so every sound has a sample in your voice.",
      "Clear and relaxed, not rushed. Slow down on the hard ones; mistakes are fine, just repeat the line.", [
          ("phon_1", "Clear", "The quick brown fox jumps over the lazy dog, while a bright yellow vixen watches from the hedge."),
          ("phon_1", "Clear", "Pack my box with five dozen liquor jugs, and how vexingly quick daft zebras jump."),
          ("phon_2", "Careful", "Red leather, yellow leather. Unique New York. The sixth sick sheik's sixth sheep's sick."),
          ("phon_2", "Careful", "Peter Piper picked a peck of pickled peppers. How much wood would a woodchuck chuck if a woodchuck could chuck wood?"),
          ("phon_3", "Soft", "Whisper words with warm, worn waves; thirty thoughtful thinkers thanked the three thin thieves."),
          ("phon_3", "Soft", "Measure the pleasure, treasure the leisure, and choose the shoes that you refuse to lose."),
          ("phon_4", "Vowels", "Beat, bit, bait, bet, bat. Boot, book, boat, bought, bot. Bite, bout, boy."),
      ]),

    S("N", "Long form", "Continuous reading", 3.0, "A few minutes of unbroken reading. Gives the longest natural stretches of pace, breath and phrasing.",
      "Read it straight through like you are telling a friend about something you care about. Do not stop for mistakes; carry on.", [
          ("long_1", "Continuous, natural", "There's a particular kind of quiet you only find early in the morning, before the city has made up its mind about the day. The streets are empty, the windows are dark, and the air smells faintly of rain and bread. It's the best time to think, and the worst time to be in a hurry."),
          ("long_2", "Continuous, natural", "I used to believe that being good at something meant never making mistakes. It took me years to understand the opposite. The people who are best at anything are simply the ones who have made the most mistakes, noticed them quickly, and kept going anyway."),
          ("long_3", "Continuous, natural", "When you learn a new skill, the first stage feels clumsy. You think about every step, you second-guess every move, and you wonder why anyone would choose this on purpose. Then, one day, without any warning, it clicks. The effort melts away, and what's left is something that feels almost like instinct."),
          ("long_4", "Continuous, natural", "Good tools, like good friends, tend to stay out of the way. They do what you ask, they surprise you only in pleasant ways, and when something goes wrong, they tell you honestly and help you fix it. That's the standard worth aiming for, whether you're building software or building a life."),
          ("long_5", "Continuous, natural", "So here is what I'd suggest. Start small. Pick one thing, do it well, and let the next thing grow out of it. Don't worry about the whole mountain. Just take the next step, and then the one after that. Before long, you'll look back and wonder how you got so far."),
      ]),
]

IDS = [s.id for s in SECTIONS]


def as_dict():
    return [{"id": s.id, "group": s.group, "title": s.title, "minutes": s.minutes, "goal": s.goal, "how": s.how,
             "legacy": s.legacy, "items": [{"tag": i.tag, "dir": i.direction, "text": i.text} for i in s.items]}
            for s in SECTIONS]


def clip_groups(section):
    """Consecutive items with the same tag -> [(tag, [item indices])]; untagged items are skipped."""
    out = []
    for k, it in enumerate(section.items):
        if not it.tag:
            continue
        if out and out[-1][0] == it.tag and out[-1][1][-1] == k - 1:
            out[-1][1].append(k)
        else:
            out.append((it.tag, [k]))
    return out


if __name__ == "__main__":
    tags = [t for s in SECTIONS if not s.legacy for t, _ in clip_groups(s)]
    assert len(tags) == len(set(tags)), [t for t in tags if tags.count(t) > 1]
    words = sum(len(i.text.split()) for s in SECTIONS for i in s.items)
    print(f"{len(SECTIONS)} sections, {len(tags)} new clip tags, {words} words, ~{sum(s.minutes for s in SECTIONS):.0f} min recorded")
