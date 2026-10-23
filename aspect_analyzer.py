"""
aspect_analyzer.py
TrustRate AI — Product Failure Root Cause & Aspect-Based Pros/Cons Diagnostic Engine

Extracts:
1. Why a certain product fails (Root Cause Diagnostics, engineering failure modes, incidence rate, severity)
2. Direct customer evidence quote snippets supporting each failure mode
3. Authentic Pros (genuine verified praise) vs Authentic Cons (real customer limitations & compromises)
4. Comprehensive Aspect Scorecard across hardware dimensions: Battery, Build Quality, Connectivity, Audio, Ergonomics
"""

import re
from typing import List, Dict, Any, Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Aspect & Failure Taxonomy
# ---------------------------------------------------------------------------

ASPECT_DEFINITIONS = {
    "battery": {
        "title": "Battery, Power & Charging",
        "icon": "🔋",
        "keywords": [
            "battery", "charge", "charges", "charging", "charger", "overheat", "overheats", "overheating",
            "hot", "warm", "drain", "draining", "drains", "cord", "cable", "plug",
            "adapter", "power", "mah", "usb-c", "lightning", "pins", "pin", "watt", "watts",
            "standby", "recharge", "runtime", "overheats dangerously", "coil whine", "electric", "short circuit"
        ],
        "failure_patterns": [
            r"\b(stopped|wont|won't|doesn't|failed to|not)\s+(charge|charging|hold a charge)\b",
            r"\b(battery|power)\s+(drains|drain|died|dies|dead|terrible|awful|horrible|poor|bad|short)\b",
            r"\b(overheat|overheats|overheating|gets\s+(very|super|dangerously)\s+hot)\b",
            r"\b(short|less than)\s+(\d+\s*(hours?|hrs?|mins?))\s+battery\b",
            r"\b(pins?|cable|plug|wire)\s+(bent|melted|loose|burned|snapped|frayed|tore)\b",
            r"\b(swollen|swelled|spark|smoke|fire\s+hazard|humming noise)\b",
            r"\b(stopped working after|died after)\s+(\d+\s*(days?|weeks?|months?))\b"
        ],
        "positive_patterns": [
            r"\b(battery|charge)\s+(lasts|lasting|life\s+is)\s+(great|amazing|long|forever|all day|hours)\b",
            r"\b(fast|quick|rapid)\s+(charging|charge)\b",
            r"\b(good|excellent|solid|impressive)\s+battery\s+life\b"
        ],
        "root_causes": {
            "overheating": "High thermal dissipation and lack of adequate heat dissipation circuitry, causing safety throttling or casing melting.",
            "charging_failure": "Pin connector wear, solder degradation on the USB port, or poor controller handshake causing charging refusal.",
            "rapid_drain": "High background power leakage, sub-par lithium cell capacity degradation within 2-4 weeks of usage.",
            "cable_defect": "Thin gauge wiring and inadequate strain relief leading to internal wire breakage at the connector junction."
        }
    },
    "build": {
        "title": "Hardware Build & Durability",
        "icon": "🛠️",
        "keywords": [
            "broke", "broken", "crack", "cracked", "cracking", "fragile", "flimsy",
            "plastic", "snap", "snapped", "fell apart", "fall apart",
            "hinge", "durability", "durable", "casing", "screws",
            "scratches", "scratched", "sturdy", "build quality", "finish", "tore", "cracking"
        ],
        "failure_patterns": [
            r"\b(broke|cracked|snapped|fell apart|shattered|tore)\s+(after|within|in)\s+(\d+\s*(days?|weeks?|months?|uses?))\b",
            r"\b(cheap|flimsy|fragile|poor)\s+(plastic|material|build|construction|quality|casing|wire)\b",
            r"\b(hinge|joint|clip|bracket|latch)\s+(broke|snapped|loose|wobbly|cracked)\b",
            r"\b(scratches?|dents?)\s+(very\s+easily|immediately|out of the box)\b",
            r"\b(pins? bent|casing cracked|wire tore)\b"
        ],
        "positive_patterns": [
            r"\b(solid|premium|sturdy|durable|well-built|well built|robust)\s+(build|feel|material|construction)\b",
            r"\b(metal|aluminum)\s+(casing|frame|finish)\b",
            r"\b(survived|dropped\s+and\s+still\s+works)\b"
        ],
        "root_causes": {
            "brittle_materials": "Use of low-grade brittle polymers without glass-fiber reinforcement, vulnerable to stress micro-fractures.",
            "premature_failure": "Lack of mechanical stress testing on hinges, joints, and connector junction strain collars.",
            "poor_tolerances": "Inconsistent injection molding tolerances resulting in creaking, loose panel gaps, and bent pin contacts."
        }
    },
    "connectivity": {
        "title": "Wireless Connectivity & Bluetooth",
        "icon": "📶",
        "keywords": [
            "bluetooth", "wireless", "disconnect", "disconnects",
            "disconnected", "disconnecting", "pair", "pairing", "paired",
            "sync", "syncing", "latency", "delay", "lag", "lagging",
            "cut out", "cutting out", "interference", "desync", "reconnect"
        ],
        "failure_patterns": [
            r"\b(keeps?|constantly|frequently|randomly)\s+(disconnecting|dropping|cutting out)\b",
            r"\b(won't|wont|unable to|fails to|refuses to)\s+(pair|connect|reconnect)\b",
            r"\b(terrible|horrible|poor|weak|short)\s+(range|signal|bluetooth)\b",
            r"\b(audio|sound)\s+(delay|lag|latency|out of sync)\b",
            r"\b(left|right)\s+(earbud|speaker|side)\s+(disconnects|loses connection|unpaired)\b"
        ],
        "positive_patterns": [
            r"\b(instant|fast|quick|seamless|easy)\s+(pairing|connection|to pair|setup)\b",
            r"\b(stable|reliable|strong|flawless)\s+(connection|bluetooth|range)\b",
            r"\b(zero|no)\s+(lag|latency|delay)\b"
        ],
        "root_causes": {
            "signal_dropout": "Sub-optimal internal antenna placement and weak RF shielding susceptible to 2.4 GHz WiFi interference.",
            "pairing_desync": "Firmware handshake timeouts between true-wireless master/slave units causing one side to drop.",
            "latency_issues": "Omission of modern low-latency Bluetooth codecs (aptX LL, AAC), producing perceptible audio-video delay."
        }
    },
    "audio": {
        "title": "Audio, Acoustic & Mic Performance",
        "icon": "🔊",
        "keywords": [
            "sound", "audio", "bass", "treble", "mids", "mic", "microphone",
            "static", "buzzing", "hiss", "hissing",
            "tinny", "muffled", "distort", "distortion", "distorted", "echo",
            "acoustic", "speaker", "driver", "call quality", "earbuds"
        ],
        "failure_patterns": [
            r"\b(static|buzzing|hissing|crackling)\s+(noise|sound|in background)\b",
            r"\b(muffled|tinny|flat|hollow|muddy|harsh)\s+(sound|audio|bass|treble|voice)\b",
            r"\b(terrible|horrible|poor|bad|awful)\s+(microphone|mic|call quality|sound quality)\b",
            r"\b(one|left|right)\s+(side|ear|driver|channel)\s+(stopped|died|silent|dead|quieter)\b",
            r"\b(max|maximum)\s+volume\s+is\s+(too\s+low|quiet|inaudible)\b",
            r"\b(distorts?|distortion)\s+at\s+(high|medium)\s+volume\b"
        ],
        "positive_patterns": [
            r"\b(crisp|clear|balanced|rich|immersive|punchy|deep)\s+(sound|audio|bass|treble|highs)\b",
            r"\b(great|excellent|superb|fantastic)\s+(sound|audio|acoustic)\s+quality\b",
            r"\b(clear|clean)\s+(microphone|mic|call quality)\b"
        ],
        "root_causes": {
            "driver_distortion": "Undersized dynamic drivers with thin diaphragms that break up under high excursion, causing audible harmonic distortion.",
            "analog_hiss": "Inadequate digital-to-analog converter (DAC) grounding producing continuous ground-loop static and hiss.",
            "poor_mic_dsp": "Absence of dual-microphone noise cancellation arrays, causing ambient wind and background clatter to overpower vocal pickup."
        }
    },
    "ergonomics": {
        "title": "Ergonomics, Comfort & Fit",
        "icon": "👂",
        "keywords": [
            "comfort", "comfortable", "uncomfortable", "fit", "fits", "fitting",
            "pain", "hurts", "hurting", "sore", "soreness", "ear tips", "tips",
            "tight", "loose", "heavy", "headband", "slips", "slipping",
            "falls out", "falling out", "cushion", "cushions", "padding"
        ],
        "failure_patterns": [
            r"\b(hurts?|painful|sore|aches?)\s+(my\s+ears?|head|after\s+\d+)\b",
            r"\b(uncomfortable|awkward|terrible)\s+(to wear|fit|design)\b",
            r"\b(keeps?|constantly)\s+(falling|slipping|dropping)\s+out\b",
            r"\b(too\s+(tight|heavy|bulky|large|small|loose))\b",
            r"\b(ear\s*tips?|cushions?)\s+(fall off|irritate|scratchy|stiff)\b"
        ],
        "positive_patterns": [
            r"\b(very|super|extremely)\s+comfortable\b",
            r"\b(comfortable|great|perfect|snug|secure)\s+fit\b",
            r"\b(lightweight|light\s+weight|hardly\s+notice|forget\s+wearing)\b"
        ],
        "root_causes": {
            "high_clamping_force": "Excessive spring tension in the headband or oversized nozzle diameter causing cartilage pressure pain within 45 minutes.",
            "slipping_fit": "Smooth medical-grade silicone tips without ergonomic concha fins, losing friction lock during jaw movement or jogging."
        }
    },
    "software": {
        "title": "Software, Controls & Compatibility",
        "icon": "📱",
        "keywords": [
            "app", "software", "controls", "touch", "button", "buttons",
            "firmware", "update", "updates", "compatible", "compatibility",
            "ios", "android", "windows", "mac", "driver", "drivers",
            "glitch", "bug", "crash", "crashes"
        ],
        "failure_patterns": [
            r"\b(app|software)\s+(crashes|glitches|useless|terrible|broken|doesn't work)\b",
            r"\b(touch\s+controls?|buttons?)\s+(unresponsive|too sensitive|accidental|don't work)\b",
            r"\b(firmware\s+update)\s+(bricked|broke|ruined)\b",
            r"\b(not|incompatible|fails)\s+with\s+(iphone|android|windows|mac)\b"
        ],
        "positive_patterns": [
            r"\b(intuitive|responsive|easy)\s+(controls|app|buttons|setup)\b",
            r"\b(app\s+has\s+(great|useful|good)\s+features|eq\s+customization)\b"
        ],
        "root_causes": {
            "capacitive_touch_mishits": "Overly sensitive capacitive surface area triggering accidental pause/skips when adjusting position in ear.",
            "unstable_companion_app": "Poorly maintained mobile companion app lacking BLE sync retry logic, causing frequent freezing."
        }
    }
}


def _clean_quote(text: str, target_keywords: Optional[List[str]] = None, max_words: int = 28) -> str:
    """Extracts a succinct sentence mentioning the target aspect keywords."""
    t = str(text).strip().replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    sentences = re.split(r"(?<=[.!?])\s+", t)
    
    if target_keywords:
        for s in sentences:
            s_low = s.lower()
            if any(re.search(r"\b" + re.escape(kw) + r"\b", s_low) for kw in target_keywords):
                s_clean = s.strip()
                if 4 <= len(s_clean.split()) <= max_words:
                    return s_clean
                elif len(s_clean.split()) > max_words:
                    return " ".join(s_clean.split()[:max_words]) + "..."

    for s in sentences:
        s_clean = s.strip()
        if 4 <= len(s_clean.split()) <= max_words:
            return s_clean
            
    words = t.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return t


def extract_aspect_sentiment_and_failures(
    reviews: Any,
    min_evidence_per_mode: int = 2
) -> Dict[str, Any]:
    """
    Core diagnostic pipeline that scans actual customer reviews and extracts:
    1. Primary Product Failure Reasons with incidence rate & root cause
    2. Real Customer Evidence quotes for every failure
    3. Authentic Pros (genuine verified praise)
    4. Authentic Cons (real customer trade-offs & limitations)
    5. Quantitative Aspect Scorecard
    """
    if isinstance(reviews, pd.DataFrame):
        rev_list = reviews.to_dict(orient="records")
    elif isinstance(reviews, list):
        rev_list = reviews
    else:
        rev_list = []

    clean_revs = []
    for r in rev_list:
        text = str(r.get("review_text", r.get("text", ""))).strip()
        if not text:
            continue
        try:
            rating = float(r.get("rating", r.get("stars", 5.0)))
        except (ValueError, TypeError):
            rating = 5.0
        
        is_susp = bool(r.get("is_suspicious", r.get("label", "") == "suspicious"))
        clean_revs.append({
            "text": text,
            "rating": rating,
            "is_suspicious": is_susp
        })

    total_reviews = len(clean_revs)
    if total_reviews == 0:
        return _empty_aspect_result()

    # Partition reviews
    critical_revs = [r for r in clean_revs if r["rating"] <= 3.0]
    genuine_high_revs = [r for r in clean_revs if r["rating"] >= 4.0 and not r["is_suspicious"]]
    if not genuine_high_revs:
        genuine_high_revs = [r for r in clean_revs if r["rating"] >= 4.0]

    # -----------------------------------------------------------------------
    # 1. Product Failure Modes Analysis (from Critical / Negative Reviews)
    # -----------------------------------------------------------------------
    failure_modes = []
    aspect_counts = {k: {"neg": 0, "pos": 0, "quotes_neg": [], "quotes_pos": []} for k in ASPECT_DEFINITIONS}

    for rev in clean_revs:
        t_low = rev["text"].lower()
        r_val = rev["rating"]

        for aspect_key, spec in ASPECT_DEFINITIONS.items():
            # Check aspect keyword presence with word boundaries
            kw_match = any(re.search(r"\b" + re.escape(kw) + r"\b", t_low) for kw in spec["keywords"])
            if not kw_match:
                continue

            # Check failure patterns
            has_fail_pattern = any(re.search(pat, t_low) for pat in spec["failure_patterns"])
            # Check positive patterns
            has_pos_pattern = any(re.search(pat, t_low) for pat in spec["positive_patterns"])

            if r_val <= 3.0 or has_fail_pattern:
                aspect_counts[aspect_key]["neg"] += 1
                q = _clean_quote(rev["text"], target_keywords=spec["keywords"])
                if q and len(aspect_counts[aspect_key]["quotes_neg"]) < 4:
                    aspect_counts[aspect_key]["quotes_neg"].append({
                        "rating": int(r_val),
                        "quote": q
                    })
            elif r_val >= 4.0 or has_pos_pattern:
                aspect_counts[aspect_key]["pos"] += 1
                q = _clean_quote(rev["text"], target_keywords=spec["keywords"])
                if q and len(aspect_counts[aspect_key]["quotes_pos"]) < 4:
                    aspect_counts[aspect_key]["quotes_pos"].append({
                        "rating": int(r_val),
                        "quote": q
                    })

    # Compile failure modes
    total_critical = max(len(critical_revs), 1)

    for aspect_key, counts in aspect_counts.items():
        neg_count = counts["neg"]
        if neg_count == 0:
            continue

        spec = ASPECT_DEFINITIONS[aspect_key]
        incidence_pct = round((neg_count / total_critical) * 100.0, 1)

        # Assign severity
        if incidence_pct >= 40.0 or any("fire" in q["quote"].lower() or "melted" in q["quote"].lower() or "overheat" in q["quote"].lower() for q in counts["quotes_neg"]):
            severity = "Critical Failure Risk"
            severity_color = "#f43f5e"  # Rose red
        elif incidence_pct >= 20.0:
            severity = "High Recurrence"
            severity_color = "#f97316"  # Orange
        else:
            severity = "Moderate Defect Rate"
            severity_color = "#eab308"  # Amber

        # Determine best root cause statement
        root_causes = spec.get("root_causes", {})
        root_cause_explanation = list(root_causes.values())[0]
        quotes_joined = " ".join(q["quote"].lower() for q in counts["quotes_neg"])
        for sub_cause, explanation in root_causes.items():
            if sub_cause.replace("_", " ") in quotes_joined:
                root_cause_explanation = explanation
                break

        failure_modes.append({
            "aspect_key": aspect_key,
            "category": spec["title"],
            "icon": spec["icon"],
            "complaint_count": neg_count,
            "incidence_pct": min(100.0, incidence_pct),
            "severity": severity,
            "severity_color": severity_color,
            "root_cause_summary": root_cause_explanation,
            "evidence_quotes": counts["quotes_neg"]
        })

    # Sort failure modes by complaint incidence descending
    failure_modes.sort(key=lambda x: (x["complaint_count"], x["incidence_pct"]), reverse=True)

    # -----------------------------------------------------------------------
    # 2. Authentic Pros Extraction (Genuine Praise)
    # -----------------------------------------------------------------------
    pros_list = []
    for aspect_key, counts in aspect_counts.items():
        pos_count = counts["pos"]
        neg_count = counts["neg"]
        if pos_count == 0:
            continue

        total_mentions = pos_count + neg_count
        sentiment_ratio = round((pos_count / max(total_mentions, 1)) * 100.0)

        if sentiment_ratio >= 50 and counts["quotes_pos"]:
            spec = ASPECT_DEFINITIONS[aspect_key]
            
            pro_titles = {
                "battery": "Reliable Power Delivery & Battery Life",
                "build": "Solid Physical Build & Structural Integrity",
                "connectivity": "Seamless Pairing & Stable Wireless Link",
                "audio": "Clear Acoustic Profile & Articulate Sound",
                "ergonomics": "Ergonomically Balanced Fit & Day-long Comfort",
                "software": "Smooth Touch Interface & Broad Compatibility"
            }
            
            best_quote = counts["quotes_pos"][0]
            pros_list.append({
                "aspect_key": aspect_key,
                "category": spec["title"],
                "icon": spec["icon"],
                "aspect_title": pro_titles.get(aspect_key, f"Commendable {spec['title']}"),
                "sentiment_score": sentiment_ratio,
                "frequency": pos_count,
                "evidence_quote": best_quote["quote"],
                "rating": best_quote["rating"]
            })

    pros_list.sort(key=lambda x: (x["sentiment_score"], x["frequency"]), reverse=True)

    if not pros_list and genuine_high_revs:
        for r in genuine_high_revs[:2]:
            pros_list.append({
                "aspect_key": "general",
                "category": "General Performance",
                "icon": "✨",
                "aspect_title": "Satisfactory General Functionality",
                "sentiment_score": 85,
                "frequency": 1,
                "evidence_quote": _clean_quote(r["text"]),
                "rating": int(r["rating"])
            })

    # -----------------------------------------------------------------------
    # 3. Authentic Cons Extraction (Real Customer Limitations)
    # -----------------------------------------------------------------------
    cons_list = []
    for mode in failure_modes:
        con_titles = {
            "battery": "Thermal Overload & Premature Battery Exhaustion",
            "build": "Vulnerable Structural Casing & Pin Wear",
            "connectivity": "Intermittent Signal Dropouts & Pairing Desync",
            "audio": "Microphone Wind Sensitivity & Static Hum",
            "ergonomics": "Ear Canal Pressure Fatigue & Slipping During Movement",
            "software": "Touch Pad Sensitivity & Companion App Latency"
        }
        evidence_quote = mode["evidence_quotes"][0]["quote"] if mode["evidence_quotes"] else "Customer reported recurring functional degradation."
        rating_quote = mode["evidence_quotes"][0]["rating"] if mode["evidence_quotes"] else 2

        cons_list.append({
            "aspect_key": mode["aspect_key"],
            "category": mode["category"],
            "icon": mode["icon"],
            "aspect_title": con_titles.get(mode["aspect_key"], f"{mode['category']} Limitations"),
            "impact_level": mode["severity"],
            "frequency": mode["complaint_count"],
            "evidence_quote": evidence_quote,
            "rating": rating_quote
        })

    # -----------------------------------------------------------------------
    # 4. Aspect Scorecard (Quantitative 0 to 100 rating for each dimension)
    # -----------------------------------------------------------------------
    scorecard = {}
    for aspect_key, spec in ASPECT_DEFINITIONS.items():
        pos = aspect_counts[aspect_key]["pos"]
        neg = aspect_counts[aspect_key]["neg"]
        total = pos + neg
        if total == 0:
            score = 70  # Baseline neutral
        else:
            score = int(round((pos / total) * 100.0))
        scorecard[spec["title"]] = {
            "score": score,
            "positive_count": pos,
            "negative_count": neg,
            "icon": spec["icon"],
            "status": "Excellent" if score >= 80 else ("Good" if score >= 60 else ("Concerning" if score >= 40 else "Defective"))
        }

    # -----------------------------------------------------------------------
    # 5. Executive Failure Summary & Risk Assessment
    # -----------------------------------------------------------------------
    primary_failure = failure_modes[0]["category"] if failure_modes else "No acute structural failure detected"
    primary_root_cause = failure_modes[0]["root_cause_summary"] if failure_modes else "Product exhibits organic consumer variance with standard wear rates."
    
    # Calculate Failure Risk Score
    if failure_modes:
        top_incidence = failure_modes[0]["incidence_pct"]
        num_defects = len(failure_modes)
        failure_risk_score = min(98, int(top_incidence * 0.65 + num_defects * 6))
    else:
        failure_risk_score = 15

    if failure_risk_score >= 70:
        risk_level = "High Risk of Failure"
        risk_color = "#f43f5e"
        durability_verdict = "Severe Hardware / Component Vulnerability Detected"
    elif failure_risk_score >= 45:
        risk_level = "Elevated Defect Likelihood"
        risk_color = "#f97316"
        durability_verdict = "Noticeable Component Wear; Proceed with Caution"
    else:
        risk_level = "Low Failure Rate"
        risk_color = "#10b981"
        durability_verdict = "Hardware Stability Meets Consumer Longevity Standards"

    executive_diagnostic = (
        f"Forensic inspection of customer reviews indicates that the primary failure driver is **{primary_failure}**, "
        f"which accounts for **{failure_modes[0]['incidence_pct'] if failure_modes else 0:.1f}%** of negative buyer feedback. "
        f"Root-cause analysis reveals: {primary_root_cause} "
        f"Overall failure risk is classified as **{risk_level}** ({failure_risk_score}/100)."
    )

    return {
        "summary": {
            "total_reviews_analyzed": total_reviews,
            "critical_reviews_count": len(critical_revs),
            "genuine_positive_count": len(genuine_high_revs),
            "primary_failure_reason": primary_failure,
            "failure_risk_score": failure_risk_score,
            "failure_risk_level": risk_level,
            "failure_risk_color": risk_color,
            "durability_verdict": durability_verdict,
            "executive_diagnostic": executive_diagnostic
        },
        "failure_modes": failure_modes,
        "pros": pros_list,
        "cons": cons_list,
        "aspect_scorecard": scorecard
    }


def _empty_aspect_result() -> Dict[str, Any]:
    return {
        "summary": {
            "total_reviews_analyzed": 0,
            "critical_reviews_count": 0,
            "genuine_positive_count": 0,
            "primary_failure_reason": "No reviews available",
            "failure_risk_score": 0,
            "failure_risk_level": "N/A",
            "failure_risk_color": "#94a3b8",
            "durability_verdict": "Insufficient data to compute diagnostic report",
            "executive_diagnostic": "No customer feedback found for this product."
        },
        "failure_modes": [],
        "pros": [],
        "cons": [],
        "aspect_scorecard": {}
    }
