"""
TB-CAD Clinical Suite — Rule-Based AI Assistant
Provides contextual clinical responses based on the latest model prediction.
No external API required — fully offline.
"""


def get_severity(prediction, confidence):
    """Determine severity level based on prediction and confidence."""
    if prediction == 'Normal':
        return 'None'
    if confidence < 70:
        return 'Low'
    elif confidence <= 85:
        return 'Moderate'
    else:
        return 'High'


def generate_response(question, prediction_context=None):
    """
    Generate a response based on the user's question and prediction context.
    
    Args:
        question: The user's question text
        prediction_context: Dict with keys: prediction, confidence, all_confidences, 
                          heatmap_path, image_name, scan_id
    
    Returns:
        response: String response text
    """
    q = question.lower().strip()
    
    # Default context values
    pred = prediction_context.get('prediction', 'Unknown') if prediction_context else 'Unknown'
    conf = prediction_context.get('confidence', 0) if prediction_context else 0
    severity = get_severity(pred, conf) if prediction_context else 'Unknown'
    all_conf = prediction_context.get('all_confidences', {}) if prediction_context else {}
    
    has_context = prediction_context is not None and pred != 'Unknown'
    
    # ── Category: Scan Meaning / What does this mean ──
    if any(kw in q for kw in ['what does this scan mean', 'what does this mean', 
                                'explain the result', 'explain result', 'interpret',
                                'what is the result', 'diagnosis', 'finding']):
        if not has_context:
            return ("No scan has been analyzed yet. Please upload a chest X-ray image on the "
                    "Chest X-Ray Analysis page first, then return here for interpretation.")
        
        if pred == 'Tuberculosis':
            return (f"📋 **Scan Interpretation — Tuberculosis Detected**\n\n"
                    f"The AI model classified this chest X-ray as **Tuberculosis** with "
                    f"**{conf:.1f}% confidence** (Severity: **{severity}**).\n\n"
                    f"This means the model detected radiographic patterns consistent with "
                    f"pulmonary tuberculosis, such as upper lobe infiltrates, cavitary lesions, "
                    f"or consolidation areas.\n\n"
                    f"**Class Probabilities:**\n"
                    f"• Normal: {all_conf.get('Normal', 0):.1f}%\n"
                    f"• Pneumonia: {all_conf.get('Pneumonia', 0):.1f}%\n"
                    f"• Tuberculosis: {all_conf.get('Tuberculosis', 0):.1f}%\n\n"
                    f"⚠️ **Important:** This is an AI-assisted screening tool. Clinical confirmation "
                    f"through GeneXpert, sputum smear microscopy, or culture is required.")
        
        elif pred == 'Pneumonia':
            return (f"📋 **Scan Interpretation — Pneumonia Detected**\n\n"
                    f"The AI model classified this chest X-ray as **Pneumonia** with "
                    f"**{conf:.1f}% confidence** (Severity: **{severity}**).\n\n"
                    f"The model identified radiographic features consistent with pneumonia, "
                    f"including areas of consolidation, ground-glass opacities, or air bronchograms.\n\n"
                    f"**Class Probabilities:**\n"
                    f"• Normal: {all_conf.get('Normal', 0):.1f}%\n"
                    f"• Pneumonia: {all_conf.get('Pneumonia', 0):.1f}%\n"
                    f"• Tuberculosis: {all_conf.get('Tuberculosis', 0):.1f}%\n\n"
                    f"🏥 Clinical follow-up with chest CT and appropriate antibiotic therapy "
                    f"assessment is recommended.")
        
        else:  # Normal
            return (f"📋 **Scan Interpretation — Normal**\n\n"
                    f"The AI model classified this chest X-ray as **Normal** with "
                    f"**{conf:.1f}% confidence**.\n\n"
                    f"No significant pathological findings were detected. The lung fields "
                    f"appear clear without evidence of consolidation, cavitation, or infiltrates.\n\n"
                    f"**Class Probabilities:**\n"
                    f"• Normal: {all_conf.get('Normal', 0):.1f}%\n"
                    f"• Pneumonia: {all_conf.get('Pneumonia', 0):.1f}%\n"
                    f"• Tuberculosis: {all_conf.get('Tuberculosis', 0):.1f}%\n\n"
                    f"✅ No immediate clinical action required based on AI analysis.")

    # ── Category: Grad-CAM Explanation ──
    elif any(kw in q for kw in ['grad-cam', 'gradcam', 'heatmap', 'explain gradcam',
                                  'activation', 'attention map', 'saliency',
                                  'highlighted region', 'what regions']):
        if not has_context:
            return ("No Grad-CAM visualization is available yet. Please analyze a chest X-ray first.")
        
        if pred == 'Normal':
            return ("🔬 **Grad-CAM Analysis — Normal Finding**\n\n"
                    "For this normal classification, the Grad-CAM heatmap shows no significant "
                    "concentrated activation in any particular lung region. This indicates "
                    "the model did not detect suspicious pathological patterns.\n\n"
                    "The diffuse, low-intensity activation pattern is consistent with a healthy "
                    "chest X-ray where no focal abnormalities draw the model's attention.")
        
        elif pred == 'Tuberculosis':
            return (f"🔬 **Grad-CAM Analysis — TB Pattern Detection**\n\n"
                    f"The Grad-CAM heatmap highlights the lung regions that most strongly "
                    f"contributed to the **Tuberculosis** classification ({conf:.1f}% confidence).\n\n"
                    f"**Key observations:**\n"
                    f"• **Red/yellow zones** indicate areas of highest model attention — these "
                    f"correspond to regions where the AI detected TB-consistent patterns.\n"
                    f"• TB typically activates the **upper lung zones**, consistent with classical "
                    f"TB pathology (apical and posterior segments).\n"
                    f"• High activation may indicate infiltrates, cavities, or consolidation.\n\n"
                    f"The heatmap uses the Grad-CAM technique on the last convolutional layer of "
                    f"EfficientNetB0 to visualize which image regions influenced the classification.")
        
        else:  # Pneumonia
            return (f"🔬 **Grad-CAM Analysis — Pneumonia Pattern Detection**\n\n"
                    f"The Grad-CAM heatmap shows areas that contributed to the **Pneumonia** "
                    f"classification ({conf:.1f}% confidence).\n\n"
                    f"**Key observations:**\n"
                    f"• **Red/yellow zones** represent high activation areas where the model "
                    f"detected pneumonia-consistent opacities.\n"
                    f"• Pneumonia patterns often appear in the **lower and middle lung zones**.\n"
                    f"• The activation pattern may correspond to consolidation or ground-glass opacities.\n\n"
                    f"This visualization helps clinicians understand and verify the AI's decision-making process.")

    # ── Category: Next Steps / What should be done ──
    elif any(kw in q for kw in ['what should be done', 'next step', 'what to do',
                                  'recommend', 'action', 'treatment', 'follow up',
                                  'what now', 'clinical action', 'management']):
        if not has_context:
            return ("Please analyze a chest X-ray first to receive clinical recommendations.")
        
        if pred == 'Tuberculosis':
            return (f"📋 **Recommended Clinical Actions — TB Detection ({severity} Severity)**\n\n"
                    f"Based on the AI classification of **Tuberculosis** at **{conf:.1f}% confidence**:\n\n"
                    f"**Immediate Actions:**\n"
                    f"1. 🧪 **GeneXpert MTB/RIF Test** — Rapid molecular test for TB DNA confirmation\n"
                    f"2. 🔬 **Sputum Smear Microscopy** — Acid-fast bacilli (AFB) staining\n"
                    f"3. 🛡️ **Airborne Isolation Precautions** — Implement infection control protocols\n"
                    f"4. 👨‍⚕️ **Pulmonology Referral** — Consult TB specialist\n\n"
                    f"**Follow-up Plan:**\n"
                    f"• Monitor symptoms (cough, fever, night sweats, weight loss)\n"
                    f"• Repeat CXR in 2 weeks for comparison\n"
                    f"• TB culture and drug sensitivity testing\n"
                    f"• Contact tracing for close contacts\n\n"
                    f"⚠️ All AI recommendations require clinical validation by a licensed physician.")
        
        elif pred == 'Pneumonia':
            return (f"📋 **Recommended Clinical Actions — Pneumonia Detection ({severity} Severity)**\n\n"
                    f"Based on the AI classification of **Pneumonia** at **{conf:.1f}% confidence**:\n\n"
                    f"**Immediate Actions:**\n"
                    f"1. 🫁 **Chest CT Follow-Up** — Detailed imaging for extent assessment\n"
                    f"2. 💊 **Antibiotic Assessment** — Empiric antibiotic therapy based on guidelines\n"
                    f"3. 📊 **Respiratory Monitoring** — SpO2, respiratory rate tracking\n"
                    f"4. 🩸 **Lab Work** — CBC, CRP, blood cultures if febrile\n\n"
                    f"**Follow-up Plan:**\n"
                    f"• Clinical reassessment in 48-72 hours\n"
                    f"• Repeat CXR in 4-6 weeks to confirm resolution\n"
                    f"• Consider atypical pathogens if no response to initial therapy")
        
        else:
            return ("✅ **No Immediate Clinical Action Required**\n\n"
                    f"The AI classified this chest X-ray as **Normal** ({conf:.1f}% confidence).\n\n"
                    f"No urgent follow-up is indicated. Standard screening protocols apply.\n\n"
                    f"If the patient has persistent symptoms despite a normal CXR, consider:\n"
                    f"• Repeat imaging in 4-6 weeks\n"
                    f"• Additional investigations (CT, sputum tests)\n"
                    f"• Clinical correlation with symptoms")

    # ── Category: Draft Report ──
    elif any(kw in q for kw in ['draft report', 'generate report', 'patient report',
                                  'clinical report', 'write report', 'create report']):
        if not has_context:
            return ("No scan data available for report generation. Please analyze a chest X-ray first.")
        
        scan_id = prediction_context.get('scan_id', 'N/A')
        image_name = prediction_context.get('image_name', 'N/A')
        
        return (f"📄 **AI-Assisted Clinical Report**\n\n"
                f"---\n"
                f"**Patient Scan ID:** {scan_id}\n"
                f"**Image:** {image_name}\n"
                f"**Date:** Current Session\n"
                f"**Modality:** Chest X-Ray (PA View)\n\n"
                f"---\n"
                f"**AI Classification:** {pred}\n"
                f"**Confidence Score:** {conf:.1f}%\n"
                f"**Severity Level:** {severity}\n\n"
                f"**Class Probabilities:**\n"
                f"• Normal: {all_conf.get('Normal', 0):.1f}%\n"
                f"• Pneumonia: {all_conf.get('Pneumonia', 0):.1f}%\n"
                f"• Tuberculosis: {all_conf.get('Tuberculosis', 0):.1f}%\n\n"
                f"**AI Interpretation:**\n"
                f"{'Radiographic patterns consistent with pulmonary tuberculosis were detected, including potential upper lobe infiltrates and/or cavitary changes.' if pred == 'Tuberculosis' else 'Radiographic features consistent with pneumonia were identified, including areas of possible consolidation or opacification.' if pred == 'Pneumonia' else 'No significant pathological findings detected. Lung fields appear clear.'}\n\n"
                f"**Model:** EfficientNetB0 (Transfer Learning)\n"
                f"**XAI Method:** Grad-CAM Visualization\n\n"
                f"---\n"
                f"⚠️ *This report is AI-generated and intended as a clinical decision-support tool. "
                f"All findings must be confirmed by a licensed radiologist or physician.*")

    # ── Category: Why TB / Reasoning ──
    elif any(kw in q for kw in ['why was tb', 'why tb', 'why tuberculosis',
                                  'why was it predicted', 'reasoning', 'how did the model',
                                  'why this prediction', 'explain prediction']):
        if not has_context:
            return ("No prediction is available to explain. Please analyze a chest X-ray first.")
        
        return (f"🧠 **Model Reasoning — {pred} Classification**\n\n"
                f"The EfficientNetB0 model was trained on a large dataset of chest X-ray images "
                f"across three classes: Normal, Pneumonia, and Tuberculosis.\n\n"
                f"**How the prediction was made:**\n"
                f"1. The uploaded image was resized to 224×224 pixels and normalized\n"
                f"2. The preprocessed image was passed through EfficientNetB0's convolutional layers\n"
                f"3. Feature extraction identified patterns in the lung regions\n"
                f"4. The softmax classifier produced probability scores for each class\n"
                f"5. The highest probability class ({pred} at {conf:.1f}%) was selected\n\n"
                f"**Confidence Breakdown:**\n"
                f"• Normal: {all_conf.get('Normal', 0):.1f}%\n"
                f"• Pneumonia: {all_conf.get('Pneumonia', 0):.1f}%\n"
                f"• Tuberculosis: {all_conf.get('Tuberculosis', 0):.1f}%\n\n"
                f"The Grad-CAM heatmap can show you exactly which regions of the image "
                f"contributed most to this prediction.")

    # ── Category: Confidence Score ──
    elif any(kw in q for kw in ['confidence', 'how confident', 'accuracy',
                                  'reliable', 'how sure', 'certainty',
                                  'probability', 'score']):
        if not has_context:
            return ("No confidence score is available. Please analyze a chest X-ray first.")
        
        reliability = "high" if conf >= 85 else "moderate" if conf >= 70 else "low"
        
        return (f"📊 **Confidence Score Analysis**\n\n"
                f"**Current Prediction:** {pred}\n"
                f"**Confidence Score:** {conf:.1f}%\n"
                f"**Reliability Level:** {reliability.capitalize()}\n\n"
                f"**Full Probability Distribution:**\n"
                f"• Normal: {all_conf.get('Normal', 0):.1f}%\n"
                f"• Pneumonia: {all_conf.get('Pneumonia', 0):.1f}%\n"
                f"• Tuberculosis: {all_conf.get('Tuberculosis', 0):.1f}%\n\n"
                f"**About the model:**\n"
                f"• Overall accuracy: 87.60%\n"
                f"• TB Precision: 97.75% (very few false positives for TB)\n"
                f"• TB Recall: 69.23% (may miss some TB cases)\n\n"
                f"{'⚠️ The confidence is below 70%. The model is uncertain about this classification. Additional clinical investigation is strongly recommended.' if conf < 70 else '✅ The model shows good confidence in this prediction, but clinical confirmation remains essential.' if conf < 85 else '✅ High confidence prediction. However, AI results should always be validated by clinical assessment.'}")

    # ── Category: Model Info ──
    elif any(kw in q for kw in ['model', 'efficientnet', 'architecture', 'how does it work',
                                  'what model', 'neural network', 'cnn', 'deep learning']):
        return ("🧠 **Model Architecture — EfficientNetB0**\n\n"
                "The TB-CAD system uses **EfficientNetB0** as its backbone neural network.\n\n"
                "**Architecture Details:**\n"
                "• **Base Model:** EfficientNetB0 pre-trained on ImageNet\n"
                "• **Approach:** Transfer Learning — the pre-trained features are fine-tuned "
                "for chest X-ray classification\n"
                "• **Input Size:** 224 × 224 × 3 (RGB)\n"
                "• **Output:** 3-class Softmax (Normal, Pneumonia, Tuberculosis)\n"
                "• **Explainability:** Grad-CAM heatmaps for visual interpretation\n\n"
                "**Performance Metrics (100-epoch model):**\n"
                "• Accuracy: 87.60%\n"
                "• Weighted F1-Score: 87.19%\n"
                "• TB Precision: 97.75%\n"
                "• TB Recall: 69.23%\n\n"
                "EfficientNetB0 uses compound scaling to balance network depth, width, and "
                "resolution, making it both efficient and accurate for medical imaging tasks.")

    # ── Category: Help / General ──
    elif any(kw in q for kw in ['help', 'what can you do', 'capabilities',
                                  'how to use', 'commands', 'options']):
        return ("🤖 **TB Diagnostic Assistant — Available Commands**\n\n"
                "I can help you with the following:\n\n"
                "1. **\"What does this scan mean?\"** — Interpret the latest prediction result\n"
                "2. **\"Explain Grad-CAM result\"** — Understand the heatmap visualization\n"
                "3. **\"What should be done next?\"** — Get clinical recommendations\n"
                "4. **\"Draft patient report\"** — Generate a clinical report\n"
                "5. **\"Why was TB predicted?\"** — Understand the model's reasoning\n"
                "6. **\"What is the confidence score?\"** — Analyze prediction reliability\n"
                "7. **\"Tell me about the model\"** — Learn about the AI architecture\n\n"
                "💡 **Tip:** First analyze a chest X-ray on the Analysis page, then ask me "
                "questions about the results for contextual responses.")

    # ── Category: Greeting ──
    elif any(kw in q for kw in ['hello', 'hi', 'hey', 'good morning', 'good evening',
                                  'greetings', 'howdy']):
        ctx = ""
        if has_context:
            ctx = (f"\n\nI see you have a recent scan result: **{pred}** at {conf:.1f}% confidence. "
                   f"Feel free to ask me about it!")
        return (f"👋 Hello, Doctor!\n\n"
                f"I'm the TB-CAD Diagnostic Assistant. I can help you interpret chest X-ray "
                f"analysis results, explain Grad-CAM heatmaps, draft clinical reports, and "
                f"provide evidence-based recommendations.{ctx}\n\n"
                f"How can I assist you today?")

    # ── Fallback ──
    else:
        suggestions = [
            "What does this scan mean?",
            "Explain Grad-CAM result",
            "What should be done next?",
            "Draft patient report",
            "Why was TB predicted?",
            "What is the confidence score?"
        ]
        
        if has_context:
            return (f"I understand you're asking about: *\"{question}\"*\n\n"
                    f"Currently, the latest scan shows **{pred}** at {conf:.1f}% confidence.\n\n"
                    f"I can help best with specific clinical questions. Try asking:\n"
                    + "\n".join(f"• \"{s}\"" for s in suggestions))
        else:
            return (f"I'm not sure how to help with that specific question.\n\n"
                    f"Here are some things I can help with:\n"
                    + "\n".join(f"• \"{s}\"" for s in suggestions)
                    + "\n\n💡 Make sure to analyze a chest X-ray first for contextual responses.")
