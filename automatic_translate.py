import pandas as pd
import re
import os

# --- Configuration ---
CSV_FILE_PATH = 'corrected_queries_WIP.csv'

# --- Enhanced translation function based on observed patterns ---
def translate_to_natural_taglish(utterance, intent, category):
    """
    Translates an English utterance to natural Taglish based on intent-specific
    and general patterns.
    """
    lower_utterance = str(utterance).lower().strip()

    # Intent-specific translations based on human correction patterns
    intent_translations = {
        'cancel_order': [
            {"pattern": r"how do i cancel my order", "result": "paano ko po ma-cancel order ko?"},
            {"pattern": r"cancel my order", "result": "i-cancel ko yung order ko"},
            {"pattern": r"need help.*cancel.*order", "result": "need ko po help ma-cancel order ko"},
            {"pattern": r"help.*cancel.*order", "result": "pa-help naman i-cancel yung order ko"},
            {"pattern": r"would it be possible.*cancel.*order", "result": "pwede ba i-cancel yung order ko?"},
            {"pattern": r"possible.*cancel.*order", "result": "pwede ba i-cancel yung order ko?"},
            {"pattern": r"problem.*cancel.*order", "result": "may problem ako sa pag-cancel ng order ko"},
            {"pattern": r"don't know how.*cancel.*order", "result": "hindi ko alam paano i-cancel yung order ko"},
            {"pattern": r"want to cancel.*order", "result": "gusto ko i-cancel yung order ko"},
            {"pattern": r"trying to cancel.*order", "result": "try ko i-cancel yung order ko"},
            {"pattern": r"assistance.*cancel.*order", "result": "need ko assistance para ma-cancel yung order ko"},
            {"pattern": r"cancelling order", "result": "cancel ko yung order"},
            {"pattern": r"cancel.*order.*made", "result": "cancel ko yung order na ginawa ko"}
        ],
        'track_order': [
            {"pattern": r"track my order", "result": "i-track ko yung order ko"},
            {"pattern": r"where.*my order", "result": "nasaan na po yung order ko?"},
            {"pattern": r"status.*order", "result": "ano na po status ng order ko?"},
            {"pattern": r"check.*order.*status", "result": "check ko lang status ng order ko"},
            {"pattern": r"order.*status", "result": "status ng order ko"},
            {"pattern": r"tracking.*order", "result": "tracking ng order ko"},
            {"pattern": r"find.*order", "result": "hanap ko yung order ko"},
            {"pattern": r"locate.*order", "result": "hanap ko yung order ko"}
        ],
        'change_order': [
            {"pattern": r"change.*order", "result": "pwede ba i-change yung order ko?"},
            {"pattern": r"modify.*order", "result": "pwede ba i-modify yung order ko?"},
            {"pattern": r"update.*order", "result": "pwede ba i-update yung order ko?"},
            {"pattern": r"edit.*order", "result": "pwede ba i-edit yung order ko?"},
            {"pattern": r"alter.*order", "result": "pwede ba i-change yung order ko?"},
            {"pattern": r"problems.*chang.*order", "result": "may problem ako sa pag-change ng something sa order ko"}
        ],
        'check_invoice': [
            {"pattern": r"check.*invoice", "result": "check ko lang yung invoice ko"},
            {"pattern": r"see.*invoice", "result": "tingnan ko yung invoice ko"},
            {"pattern": r"view.*invoice", "result": "tingnan ko yung invoice ko"},
            {"pattern": r"invoice.*last month", "result": "pacheck po ng invoice last month"},
            {"pattern": r"download.*invoice", "result": "download ko yung invoice ko"},
            {"pattern": r"get.*invoice", "result": "kunin ko yung invoice ko"},
            {"pattern": r"checking invoice", "result": "chine-check ko lang yung invoice"}
        ],
        'get_refund': [
            {"pattern": r"get.*refund", "result": "paano po makaka-get ng refund?"},
            {"pattern": r"request.*refund", "result": "paano po mag-request ng refund?"},
            {"pattern": r"refund.*order", "result": "pwede ba ma-refund yung order ko?"},
            {"pattern": r"want.*refund", "result": "gusto ko ng refund"},
            {"pattern": r"need.*refund", "result": "need ko ng refund"},
            {"pattern": r"how.*refund", "result": "paano po yung refund?"},
            {"pattern": r"return.*money", "result": "pwede ba ibalik yung bayad ko?"}
        ],
        'contact_customer_service': [
            {"pattern": r"contact.*customer.*service", "result": "paano po makakontact ng customer service?"},
            {"pattern": r"talk.*customer.*service", "result": "paano po makakausap customer service?"},
            {"pattern": r"speak.*customer.*service", "result": "paano po makakausap customer service?"},
            {"pattern": r"reach.*customer.*service", "result": "paano po maabot customer service?"},
            {"pattern": r"call.*customer.*service", "result": "paano po tumawag sa customer service?"}
        ],
        'contact_human_agent': [
            {"pattern": r"talk.*human", "result": "makakausap ba ako ng human agent?"},
            {"pattern": r"speak.*human", "result": "makakausap ba ako ng human?"},
            {"pattern": r"human.*agent", "result": "pwede ba makausap yung human agent?"},
            {"pattern": r"real person", "result": "pwede ba makausap yung real person?"}
        ],
        'check_payment_methods': [
            {"pattern": r"payment.*method", "result": "ano po yung mga payment methods?"},
            {"pattern": r"how.*pay", "result": "paano po magbayad?"},
            {"pattern": r"payment.*option", "result": "ano po yung payment options?"},
            {"pattern": r"ways to pay", "result": "ano po yung paraan ng pagbayad?"}
        ],
        'delivery_period': [
            {"pattern": r"delivery.*time", "result": "gaano po katagal yung delivery?"},
            {"pattern": r"when.*deliver", "result": "kailan po idedeliver?"},
            {"pattern": r"how long.*delivery", "result": "gaano po katagal yung delivery?"},
            {"pattern": r"delivery.*period", "result": "gaano po katagal yung delivery period?"},
            {"pattern": r"shipping.*time", "result": "gaano po katagal yung shipping?"}
        ],
        'change_shipping_address': [
            {"pattern": r"change.*shipping.*address", "result": "pwede ba i-change yung shipping address?"},
            {"pattern": r"update.*address", "result": "may problema po ako sa pag-update ng address ko"},
            {"pattern": r"correct.*delivery.*address", "result": "pa-help naman, mali yung delivery address. paano ba ayusin to?"},
            {"pattern": r"wrong.*address", "result": "mali yung address ko, paano ba i-correct?"},
            {"pattern": r"delivery.*address", "result": "delivery address ko"}
        ],
        'check_cancellation_fee': [
            {"pattern": r"cancellation.*charge", "result": "check ko lang sana yung cancellation charge"},
            {"pattern": r"cancellation.*fee", "result": "magkano po yung cancellation fee?"},
            {"pattern": r"check.*cancellation.*charge", "result": "gusto ko ng tulong para icheck yung charge sa cancellation"},
            {"pattern": r"wanna check.*cancellation", "result": "check ko lang sana yung cancellation charge"}
        ],
        'place_order': [
            {"pattern": r"place.*order", "result": "paano po mag-place ng order?"},
            {"pattern": r"make.*order", "result": "paano po gumawa ng order?"},
            {"pattern": r"create.*order", "result": "paano po gumawa ng order?"},
            {"pattern": r"submit.*order", "result": "paano po i-submit yung order?"}
        ],
        'create_account': [
            {"pattern": r"create.*account", "result": "paano po gumawa ng account?"},
            {"pattern": r"make.*account", "result": "paano po gumawa ng account?"},
            {"pattern": r"sign up", "result": "paano po mag-sign up?"},
            {"pattern": r"register", "result": "paano po mag-register?"}
        ],
        'delete_account': [
            {"pattern": r"delete.*account", "result": "paano po i-delete yung account ko?"},
            {"pattern": r"remove.*account", "result": "paano po i-remove yung account ko?"},
            {"pattern": r"close.*account", "result": "paano po i-close yung account ko?"}
        ],
        'payment_issue': [
            {"pattern": r"payment.*problem", "result": "may payment problem ako"},
            {"pattern": r"payment.*issue", "result": "may payment issue ako"},
            {"pattern": r"payment.*error", "result": "may payment error"},
            {"pattern": r"problem.*payment", "result": "may problem sa payment ko"}
        ],
        'complaint': [
            {"pattern": r"complaint", "result": "may complaint ako"},
            {"pattern": r"complain", "result": "mag-complain ako"},
            {"pattern": r"report.*problem", "result": "i-report ko yung problem"},
            {"pattern": r"file.*complaint", "result": "mag-file ako ng complaint"}
        ]
    }

    # Try intent-specific patterns first
    if intent in intent_translations:
        for trans in intent_translations[intent]:
            if re.search(trans["pattern"], lower_utterance):
                return trans["result"]

    # General patterns for common sentence structures
    general_patterns = [
        {"pattern": r"^how do i (.+)", "result": r"paano ko po \1?"},
        {"pattern": r"^i need help with (.+)", "result": r"need ko po help sa \1"},
        {"pattern": r"^i need help (.+)", "result": r"need ko po help \1"},
        {"pattern": r"^can you help me (.+)", "result": r"pwede ba tulungan mo ako \1?"},
        {"pattern": r"^help me (.+)", "result": r"tulungan mo ako \1"},
        {"pattern": r"^i want to (.+)", "result": r"gusto ko po \1"},
        {"pattern": r"^i would like to (.+)", "result": r"gusto ko po \1"},
        {"pattern": r"^is it possible to (.+)", "result": r"pwede ba \1?"},
        {"pattern": r"^can i (.+)", "result": r"pwede ba ako \1?"},
        {"pattern": r"^i have a problem with (.+)", "result": r"may problem ako sa \1"},
        {"pattern": r"^i have issues with (.+)", "result": r"may problema ako sa \1"},
        {"pattern": r"^problem with (.+)", "result": r"may problem sa \1"},
        {"pattern": r"^where is (.+)", "result": r"nasaan po yung \1?"},
        {"pattern": r"^when will (.+)", "result": r"kailan po \1?"},
        {"pattern": r"^what is (.+)", "result": r"ano po yung \1?"},
        {"pattern": r"^why (.+)", "result": r"bakit po \1?"},
        {"pattern": r"^wanna (.+)", "result": r"gusto ko lang \1"},
        {"pattern": r"^want (.+)", "result": r"gusto ng \1"},
        {"pattern": r"^trying to (.+)", "result": r"sinusubukan ko \1"}
    ]

    for pattern in general_patterns:
        match = re.match(pattern["pattern"], lower_utterance)
        if match:
            result = re.sub(pattern["pattern"], pattern["result"], utterance, flags=re.IGNORECASE)
            # Clean up common issues
            result = re.sub(r"\bmy\b", "ko", result, flags=re.IGNORECASE)
            result = re.sub(r"\bthe\b", "yung", result, flags=re.IGNORECASE)
            result = re.sub(r"\border\b", "order", result, flags=re.IGNORECASE)
            result = re.sub(r"\baccount\b", "account", result, flags=re.IGNORECASE)
            result = re.sub(r"\binvoice\b", "invoice", result, flags=re.IGNORECASE)
            result = re.sub(r"\brefund\b", "refund", result, flags=re.IGNORECASE)
            result = re.sub(r"\bcancel\b", "cancel", result, flags=re.IGNORECASE)
            result = re.sub(r"\btrack\b", "track", result, flags=re.IGNORECASE)
            return result

    # Simple word-based fallback for short phrases
    result = utterance
    word_replacements = {
        'my order': 'order ko',
        'the order': 'yung order',
        'my account': 'account ko',
        'the account': 'yung account',
        'cancel order': 'cancel ng order',
        'track order': 'track ng order',
        'help me': 'tulungan mo ako',
        'thank you': 'salamat po'
    }

    for eng, tag in word_replacements.items():
        result = re.sub(r'\b' + re.escape(eng) + r'\b', tag, result, flags=re.IGNORECASE)

    # If still unchanged, return a simple transformation
    if result == utterance:
        result = re.sub(r"my", "yung", utterance, flags=re.IGNORECASE)
        result = re.sub(r"the", "yung", result, flags=re.IGNORECASE).lower()

    return result

# --- Main Script ---
def main():
    # Read the CSV file
    try:
        parsed_data = pd.read_csv(CSV_FILE_PATH)
        print(f"Loaded {len(parsed_data)} rows")
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Ensure necessary columns exist
    required_columns = ['utterance', 'intent', 'category', 'human_corrected_tagalog', 'tagalog']
    for col in required_columns:
        if col not in parsed_data.columns:
            # Add missing columns with NaN or empty string as default
            if col == 'human_corrected_tagalog':
                parsed_data[col] = ''
            else:
                parsed_data[col] = None
            print(f"Warning: Column '{col}' not found. Added with default values.")

    # Convert columns to string type to avoid errors with .strip()
    parsed_data['utterance'] = parsed_data['utterance'].astype(str)
    parsed_data['intent'] = parsed_data['intent'].astype(str)
    parsed_data['category'] = parsed_data['category'].astype(str)
    parsed_data['human_corrected_tagalog'] = parsed_data['human_corrected_tagalog'].fillna('').astype(str)
    parsed_data['tagalog'] = parsed_data['tagalog'].fillna('').astype(str) # For original machine translation

    # Find rows that need translation
    rows_without_corrections_mask = parsed_data['human_corrected_tagalog'].apply(lambda x: pd.isna(x) or str(x).strip() == '')
    rows_without_corrections = parsed_data[rows_without_corrections_mask].copy()

    print(f"Found {len(rows_without_corrections)} rows without corrections")

    # Process all rows without corrections
    print('Generating translations...')
    translated_rows = []
    for index, row in rows_without_corrections.iterrows():
        if index % 500 == 0:
            print(f"Processed {index} rows...")

        translation = translate_to_natural_taglish(row['utterance'], row['intent'], row['category'])
        translated_rows.append({
            'original_index': index, # Store original index to update DataFrame
            'utterance': row['utterance'],
            'intent': row['intent'],
            'category': row['category'],
            'human_corrected_tagalog': translation,
            'tagalog': row['tagalog'] # Keep original machine translation
        })

    print(f"Generated {len(translated_rows)} new translations")

    # Update the original DataFrame with new translations
    for translated_row in translated_rows:
        original_index = translated_row['original_index']
        parsed_data.loc[original_index, 'human_corrected_tagalog'] = translated_row['human_corrected_tagalog']

    # Final statistics
    final_rows_with_corrections = parsed_data[
        parsed_data['human_corrected_tagalog'].apply(lambda x: pd.notna(x) and str(x).strip() != '')
    ]

    print("\n=== RESULTS ===")
    print(f"Original rows with corrections: (This value needs to be manually observed from your dataset, assuming 198 from JS code)")
    print(f"New translations generated: {len(translated_rows)}")
    print(f"Total rows with corrections now: {len(final_rows_with_corrections)}")
    print(f"Remaining rows without corrections: {len(parsed_data) - len(final_rows_with_corrections)}")

    # Show some examples
    print("\n=== SAMPLE TRANSLATIONS ===")
    # To ensure we show newly translated samples, we can filter for them
    sample_display_count = 15
    displayed_count = 0
    for index, row in parsed_data.iterrows():
        if rows_without_corrections_mask.iloc[index] and displayed_count < sample_display_count:
            print(f"{displayed_count + 1}. [{row['intent']}]")
            print(f"   English: \"{row['utterance']}\"")
            print(f"   Generated: \"{row['human_corrected_tagalog']}\"")
            print(f"   Original machine: \"{row['tagalog']}\"")
            print("---")
            displayed_count += 1
        if displayed_count >= sample_display_count:
            break


    # Convert to CSV
    output_csv_path = 'updated_corrected_queries_WIP.csv'
    parsed_data.to_csv(output_csv_path, index=False, encoding='utf-8', quoting=1) # quoting=1 for QUOTE_ALL

    print("\n=== CSV READY ===")
    print(f"Total rows in updated CSV: {len(parsed_data)}")
    print(f"Updated CSV with new translations saved to '{output_csv_path}'")

if __name__ == "__main__":
    main()