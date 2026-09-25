import json
from google import genai
from pydantic import BaseModel, Field
from google.cloud import translate_v2 as translate

# ==========================================
# Day 16: Interoperability Configuration
# ==========================================
# This dictionary represents the contents of country_config.json[cite: 1, 2]
india_config = {
    "country": "India",
    "alert_language": "hi",
    "thresholds": [
        {"max": 100, "label": "Satisfactory/Moderate", "trigger_alert": False},
        {"max": 200, "label": "Moderate", "trigger_alert": False},
        {"max": 300, "label": "Poor", "trigger_alert": True},
        {"max": 400, "label": "Very Poor", "trigger_alert": True},
        {"max": float('inf'), "label": "Severe", "trigger_alert": True}
    ]
}

# ==========================================
# Day 12: Dynamic Threshold Logic
# ==========================================
def classify_aqi_dynamic(aqi_value: float, config: dict) -> tuple[str, bool]:
    """
    Evaluates predicted AQI against a dynamically loaded country configuration 
    to determine severity and alert status[cite: 1, 2].
    """
    for band in config["thresholds"]:
        if aqi_value <= band["max"]:
            return band["label"], band["trigger_alert"]
    return "Unknown", False

# ==========================================
# Day 13: Gemini Agent Drafting
# ==========================================
class AlertDraft(BaseModel):
    district: str = Field(description="The affected district or corridor")
    severity: str = Field(description="The official AQI severity category")
    likely_cause: str = Field(description="The primary contributing cause inferred from feature data")
    recommended_actions: list[str] = Field(description="Two concrete, short-term preventive actions")

def generate_alert(district: str, aqi: float, severity: str, features: dict) -> AlertDraft:
    """
    Uses Gemini structured outputs to draft a localized natural-language alert 
    with cause and recommended action[cite: 1, 2].
    """
    client = genai.Client()
    
    prompt = f"""
    You are an environmental risk advisor for local authorities. Analyze this 48-hour forecast:
    - District: {district}
    - Predicted AQI: {aqi} ({severity})
    - Feature Importances: {json.dumps(features)}
    
    Determine the likely cause (e.g., wind stagnation vs. agricultural burning) and recommend 2 targeted interventions[cite: 2].
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": AlertDraft,
        }
    )
    return AlertDraft.model_validate_json(response.text)

# ==========================================
# Day 14: Translation / Localization
# ==========================================
def translate_alert(text: str, target_lang: str) -> str:
    """
    Translates the plain-language English alert into the target language 
    specified in the configuration file[cite: 1, 2].
    """
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_lang)
    return result["translatedText"]

# ==========================================
# Day 15: End-to-End Orchestration
# ==========================================
def run_phase_4_pipeline(forecast_row: dict, config: dict) -> dict | None:
    """
    Wires the agent pipeline: Threshold -> Gemini -> Translation -> Output payload[cite: 1, 2].
    """
    # 1. Dynamic Threshold Check
    severity, is_breached = classify_aqi_dynamic(forecast_row["predicted_aqi"], config)
    if not is_breached:
        return None

    # 2. Gemini Draft Generation
    draft_obj = generate_alert(
        district=forecast_row["district"],
        aqi=forecast_row["predicted_aqi"],
        severity=severity,
        features=forecast_row.get("feature_importances", {})
    )
    
    english_alert = (
        f"URGENT ALERT for {draft_obj.district}: "
        f"AQI is forecasted to reach {draft_obj.severity} levels in 48-72 hours. "
        f"Likely cause: {draft_obj.likely_cause}. "
        f"Recommended actions: {', '.join(draft_obj.recommended_actions)}."
    )
    
    # 3. Dynamic Translation
    localized_alert = translate_alert(english_alert, config["alert_language"])
    
    # 4. Final Routed Payload
    return {
        "district": draft_obj.district,
        "forecasted_aqi": forecast_row["predicted_aqi"],
        "severity": severity,
        "alert_english": english_alert,
        f"alert_{config['alert_language']}": localized_alert,
    }

# ==========================================
# Execution / Test Case
# ==========================================
if __name__ == "__main__":
    # Mock data output representing the Vertex AI / fallback model forecast[cite: 1, 2]
    mock_bq_forecast = {
        "district": "Anand Vihar",
        "predicted_aqi": 345,
        "feature_importances": {"fire_count": 0.65, "wind_u": 0.1, "temp": 0.05}
    }
    
    # Run the pipeline with the mock data and India configuration
    alert_payload = run_phase_4_pipeline(mock_bq_forecast, india_config)
    
    if alert_payload:
        print(json.dumps(alert_payload, indent=2, ensure_ascii=False))