from writing_assistance import FormalityDetector


def main():
    detector_name = input("Enter the name of the model ('deberta', 'xlm_roberta', 'gpt'): ").lower()
    text = input('Enter the text for formality detection: ')
    try:
        formality_score = FormalityDetector.predict(detector_name, text)
        print(f"Formality score for the text using the '{detector_name}' model: {formality_score}")
    except ValueError as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
