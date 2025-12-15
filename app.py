from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import traceback
import os

# Import your existing backend code
try:
    # Try to import transformers - handle if not available
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch
    import zlib
    import numpy as np
    import plotly.graph_objects as go
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers library not available. Using mock data.")

app = Flask(__name__)
CORS(app)

# Your existing code adapted as functions
class PlantUMLEncoder:
    @staticmethod
    def encode(text: str) -> str:
        zlibbed = zlib.compress(text.encode("utf-8"))[2:-4]
        
        def encode6bit(b: int) -> str:
            if b < 10: return chr(48 + b)
            b -= 10
            if b < 26: return chr(65 + b)
            b -= 26
            if b < 26: return chr(97 + b)
            b -= 26
            return "-" if b == 0 else "_" if b == 1 else "?"
        
        encoded = ""
        for i in range(0, len(zlibbed), 3):
            b1 = zlibbed[i]
            b2 = zlibbed[i + 1] if i + 1 < len(zlibbed) else 0
            b3 = zlibbed[i + 2] if i + 2 < len(zlibbed) else 0
            
            c1 = b1 >> 2
            c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
            c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
            c4 = b3 & 0x3F
            
            encoded += (
                encode6bit(c1) +
                encode6bit(c2) +
                encode6bit(c3) +
                encode6bit(c4)
            )
        
        return encoded
    
    @staticmethod
    def make_url(uml_text: str) -> str:
        return "https://www.plantuml.com/plantuml/svg/" + PlantUMLEncoder.encode(uml_text)

class MockBPMNExtractor:
    """Mock extractor for when transformers is not available"""
    
    COLOR_MAP = {
        "AGENT": "#7FDBFF",
        "TASK": "#FFDC00",
        "CONDITION": "#FF851B",
        "TASK_INFO": "#2ECC40",
        "O": "#DDDDDD",
    }
    
    @staticmethod
    def build_bpmn(text: str):
        # Mock structure based on input text
        entities = [
            ("The customer", "B-AGENT"),
            ("submits a refund request", "B-TASK"),
            ("the support agent", "B-AGENT"),
            ("reviews the request", "B-TASK"),
            ("if the request is valid", "B-CONDITION"),
            ("the finance department", "B-AGENT"),
            ("approves the refund", "B-TASK"),
            ("the system", "B-AGENT"),
            ("processes the payment", "B-TASK"),
            ("the customer", "B-AGENT"),
            ("receives a confirmation email", "B-TASK"),
            ("else", "O"),
            ("the support agent", "B-AGENT"),
            ("informs the customer", "B-TASK"),
        ]
        
        # Create entity HTML
        html_parts = []
        for word, label in entities:
            base = label.split("-")[-1] if "-" in label else label
            color = MockBPMNExtractor.COLOR_MAP.get(base, "#ccc")
            html_parts.append(f'<span class="entity entity-{base.lower() if base != "O" else "other"}">{word}</span>')
        entity_html = " ".join(html_parts)
        
        # Create PlantUML
        plantuml = """@startuml
|The customer|
:submits a refund request;
|the support agent|
:reviews the request;
if (if the request is valid?) then (yes)
|the finance department|
:approves the refund;
|the system|
:processes the payment;
|the customer|
:receives a confirmation email;
else (no)
|the support agent|
:informs the customer;
endif
stop
@enduml"""
        
        plantuml_url = PlantUMLEncoder.make_url(plantuml)
        
        # Mock agents and matrix
        agents = ['The customer', 'the support agent', 'the finance department', 'the system']
        matrix = [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        # Mock sankey data
        sankey_sources = [0, 1, 2, 3]
        sankey_targets = [1, 2, 3, 0]
        sankey_values = [1, 1, 1, 1]
        
        return {
            "entities": entities,
            "entity_html": entity_html,
            "plantuml": plantuml,
            "plantuml_url": plantuml_url,
            "agents": agents,
            "matrix": matrix,
            "sankey_sources": sankey_sources,
            "sankey_targets": sankey_targets,
            "sankey_values": sankey_values
        }

# Initialize the extractor
if TRANSFORMERS_AVAILABLE:
    try:
        from dataclasses import dataclass
        from typing import List, Tuple, Dict, Any
        
        @dataclass
        class BPMNConfig:
            model_name: str = "jtlicardo/bpmn-information-extraction-v2"
        
        class BPMNExtractor:
            def __init__(self, cfg: BPMNConfig):
                self.cfg = cfg
                self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(cfg.model_name)
                self.color_map = {
                    "AGENT": "#7FDBFF",
                    "TASK": "#FFDC00",
                    "CONDITION": "#FF851B",
                    "TASK_INFO": "#2ECC40",
                    "O": "#DDDDDD",
                }
            
            def extract_entities(self, text: str) -> List[Tuple[str, str]]:
                inputs = self.tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                predictions = torch.argmax(outputs.logits, dim=-1)[0]
                
                merged: List[Tuple[str, str]] = []
                current_word = ""
                current_label = None
                
                for token, pred in zip(tokens, predictions):
                    label = self.model.config.id2label[int(pred)]
                    
                    if token.startswith("##"):
                        current_word += token[2:]
                        continue
                    
                    if current_word:
                        merged.append((current_word, current_label))
                    
                    current_word = token
                    current_label = label
                
                if current_word:
                    merged.append((current_word, current_label))
                
                return merged
            
            def entities_to_structure(self, entities: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
                structure: List[Dict[str, Any]] = []
                cur_word = ""
                cur_type = None
                
                for word, label in entities:
                    low = word.lower()
                    if low in ["else", "otherwise"]:
                        if cur_word:
                            structure.append({"word": cur_word.strip(), "type": cur_type})
                        structure.append({"word": low, "type": "CONDITION"})
                        cur_word = ""
                        cur_type = None
                        continue
                    
                    base = label.split("-")[-1] if "-" in label else label
                    
                    if base in ["AGENT", "TASK", "CONDITION", "TASK_INFO"]:
                        if label.startswith("B-"):
                            if cur_word:
                                structure.append({"word": cur_word.strip(), "type": cur_type})
                            cur_word = word + " "
                            cur_type = base
                        elif label.startswith("I-") and cur_type == base:
                            cur_word += word + " "
                    else:
                        if cur_word:
                            structure.append({"word": cur_word.strip(), "type": cur_type})
                        cur_word = ""
                        cur_type = None
                
                if cur_word:
                    structure.append({"word": cur_word.strip(), "type": cur_type})
                
                return structure
            
            def structure_to_bpmn(self, structure: List[Dict[str, Any]]) -> str:
                bpmn = ["@startuml"]
                current_lane = None
                inside_if = False
                
                for item in structure:
                    word = item["word"]
                    t = item["type"]
                    
                    if t == "AGENT":
                        if word != current_lane:
                            bpmn.append(f"|{word}|")
                            current_lane = word
                    
                    elif t == "CONDITION":
                        if word.lower() in ["else", "otherwise"]:
                            bpmn.append("else (no)")
                        else:
                            bpmn.append(f"if ({word}?) then (yes)")
                            inside_if = True
                    
                    elif t == "TASK":
                        bpmn.append(f":{word};")
                
                if inside_if:
                    bpmn.append("endif")
                
                bpmn.append("stop")
                bpmn.append("@enduml")
                return "\n".join(bpmn)
            
            def create_entity_html(self, entities: List[Tuple[str, str]]) -> str:
                grouped = []
                cur = ""
                cur_label = None
                
                for word, label in entities:
                    base = label.split("-")[-1] if "-" in label else label
                    if base in ["AGENT", "TASK", "CONDITION", "TASK_INFO"]:
                        if label.startswith("B-"):
                            if cur:
                                grouped.append((cur.strip(), cur_label))
                            cur = word + " "
                            cur_label = base
                        elif label.startswith("I-") and cur_label == base:
                            cur += word + " "
                    else:
                        if cur:
                            grouped.append((cur.strip(), cur_label))
                            cur = ""
                            cur_label = None
                        grouped.append((word, "O"))
                
                if cur:
                    grouped.append((cur.strip(), cur_label))
                
                html_parts = []
                for word, lbl in grouped:
                    base = lbl.split("-")[-1] if "-" in lbl else lbl
                    color_class = f"entity-{base.lower()}" if base != "O" else "entity-other"
                    html_parts.append(f'<span class="entity {color_class}">{word}</span>')
                
                return " ".join(html_parts)
            
            def build_bpmn(self, text: str) -> Dict[str, Any]:
                entities = self.extract_entities(text)
                structure = self.entities_to_structure(entities)
                plantuml = self.structure_to_bpmn(structure)
                plantuml_url = PlantUMLEncoder.make_url(plantuml)
                entity_html = self.create_entity_html(entities)
                
                # Resource analyzer functionality
                agents = [x["word"] for x in structure if x["type"] == "AGENT"]
                freq = {}
                for i in range(len(agents) - 1):
                    a, b = agents[i], agents[i + 1]
                    if a != b:
                        freq[(a, b)] = freq.get((a, b), 0) + 1
                
                # Create matrix
                unique_agents = sorted(set(agents))
                idx = {a: i for i, a in enumerate(unique_agents)}
                matrix = np.zeros((len(unique_agents), len(unique_agents)))
                
                for (a1, a2), value in freq.items():
                    if a1 in idx and a2 in idx:
                        matrix[idx[a1]][idx[a2]] = value
                
                # Create sankey data
                sankey_sources = []
                sankey_targets = []
                sankey_values = []
                
                for (a, b), v in freq.items():
                    if a in idx and b in idx:
                        sankey_sources.append(idx[a])
                        sankey_targets.append(idx[b])
                        sankey_values.append(v)
                
                return {
                    "entities": entities,
                    "entity_html": entity_html,
                    "plantuml": plantuml,
                    "plantuml_url": plantuml_url,
                    "agents": unique_agents,
                    "matrix": matrix.tolist(),
                    "sankey_sources": sankey_sources,
                    "sankey_targets": sankey_targets,
                    "sankey_values": sankey_values
                }
        
        # Initialize real extractor
        config = BPMNConfig()
        extractor = BPMNExtractor(config)
        print("Real BPMN extractor initialized successfully")
        
    except Exception as e:
        print(f"Error initializing real extractor: {e}")
        TRANSFORMERS_AVAILABLE = False
        extractor = MockBPMNExtractor()
else:
    extractor = MockBPMNExtractor()
    print("Using mock BPMN extractor")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_text():
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Process the text using the extractor
        result = extractor.build_bpmn(text)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing text: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route('/test')
def test_endpoint():
    return jsonify({"status": "ok", "message": "Server is running"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)