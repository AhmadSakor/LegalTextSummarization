import json
from rdflib import Graph, Literal, RDF, URIRef, Namespace, XSD, RDFS
import os
import re
from urllib.parse import quote
import dateparser
import logging
from hashlib import md5
import argparse

# Define RDF namespaces for Legal Case and Entity data
LEGAL = Namespace("http://example.org/legalcase#")
ENTITY = Namespace("http://example.org/entity#")

def sanitize_uri(uri):
    """Sanitizes a URI string to make it compatible with RDF format."""
    return quote(re.sub(r'[^\w\-_\.]', '_', uri), safe='')

def sanitize_property(prop):
    """Sanitizes a property name by converting spaces to underscores and lowering case."""
    return re.sub(r'\s+', '_', prop.lower())

def convert_arabic_date(date_str):
    """Converts an Arabic date string into a standardized format (YYYY-MM-DD)."""
    if not date_str or date_str in ['<undefined>', 'دون تاريخ', 'محدد-01-تاريخ', 'سابق-01-06']:
        return None
    
    try:
        parsed_date = dateparser.parse(date_str, languages=['ar'])
        if parsed_date:
            return parsed_date.strftime('%Y-%m-%d')
    except Exception as e:
        logging.error(f"Date parsing error for '{date_str}': {str(e)}")
    
    return date_str

def safe_add(graph, subject, predicate, obj):
    """Safely adds a triple to an RDF graph, with error handling to log any issues."""
    try:
        graph.add((subject, predicate, obj))
    except Exception as e:
        logging.error(f"Error adding triple: ({subject}, {predicate}, {obj}). Error: {str(e)}")

def create_entity(graph, entity_type, value):
    """Creates an RDF entity with a unique identifier based on the hashed value."""
    entity_id = md5(value.encode()).hexdigest()
    entity_uri = URIRef(ENTITY + entity_type + '_' + entity_id)
    
    safe_add(graph, entity_uri, RDF.type, getattr(ENTITY, entity_type))
    safe_add(graph, entity_uri, ENTITY.value, Literal(value))
    safe_add(graph, entity_uri, RDFS.label, Literal(value))  # Add label for entity
    
    return entity_uri

def json_to_rdf(json_file, ontology_graph):
    """Converts a JSON file of legal case data into RDF triples."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    g = Graph()  # Initialize a new graph
    g += ontology_graph  # Add ontology graph to the case graph

    # Bind prefixes for namespaces
    g.bind("legal", LEGAL)
    g.bind("entity", ENTITY)

    # Process case information
    case_number = sanitize_uri(data.get('case_information', {}).get('case_number', 'unknown'))
    case = URIRef(LEGAL + case_number)
    safe_add(g, case, RDF.type, LEGAL.LegalCase)
    safe_add(g, case, RDFS.label, Literal(f"Case {case_number}"))  # Add label for case

    # Add full text of the case if available
    if 'full_text' in data:
        safe_add(g, case, LEGAL.fullText, Literal(data['full_text']))

    # Handle case information details
    if 'case_information' in data:
        case_info = URIRef(case + "_info")
        safe_add(g, case, LEGAL.hasCaseInformation, case_info)
        safe_add(g, case_info, RDF.type, LEGAL.CaseInformation)
        safe_add(g, case_info, RDFS.label, Literal(f"Case Information for {case_number}"))  # Add label
        
        for key, value in data['case_information'].items():
            # Handle date of ruling with Arabic date parsing
            if key == 'date_of_ruling':
                date_of_ruling = convert_arabic_date(value)
                if date_of_ruling:
                    if re.match(r'\d{4}-\d{2}-\d{2}', date_of_ruling):
                        safe_add(g, case_info, LEGAL.dateOfRuling, Literal(date_of_ruling, datatype=XSD.date))
                    else:
                        safe_add(g, case_info, LEGAL.dateOfRuling, Literal(date_of_ruling))
            # Create entities for court and main case topic
            elif key in ['court', 'main_case_topic']:
                entity = create_entity(g, sanitize_property(key), value)
                safe_add(g, case_info, getattr(LEGAL, sanitize_property(key)), entity)
            else:
                safe_add(g, case_info, getattr(LEGAL, sanitize_property(key)), Literal(value))

    # Handle persons involved in the case
    if 'persons_involved' in data:
        for person in data['persons_involved']:
            person_uri = URIRef(case + "_person_" + sanitize_uri(person.get('name', 'unknown')))
            safe_add(g, case, LEGAL.hasPerson, person_uri)
            safe_add(g, person_uri, RDF.type, LEGAL.Person)
            safe_add(g, person_uri, RDFS.label, Literal(person.get('name', 'Unknown Person')))  # Add label
            for key, value in person.items():
                safe_add(g, person_uri, getattr(LEGAL, sanitize_property(key)), Literal(value))

    # Handle background information
    if 'background_of_the_case' in data:
        background = URIRef(case + "_background")
        safe_add(g, case, LEGAL.hasBackgroundOfTheCase, background)
        safe_add(g, background, RDF.type, LEGAL.BackgroundOfTheCase)
        safe_add(g, background, RDFS.label, Literal(f"Background of Case {case_number}"))  # Add label
        
        if 'overview' in data['background_of_the_case']:
            safe_add(g, background, LEGAL.overview, Literal(data['background_of_the_case']['overview']))
        
        if 'relevant_dates' in data['background_of_the_case']:
            for date_info in data['background_of_the_case']['relevant_dates']:
                date_uri = URIRef(background + "_date_" + sanitize_uri(date_info.get('date', 'unknown')))
                safe_add(g, background, LEGAL.hasRelevantDate, date_uri)
                safe_add(g, date_uri, RDF.type, LEGAL.RelevantDate)
                safe_add(g, date_uri, RDFS.label, Literal(f"Date: {date_info.get('date', 'Unknown')}"))  # Add label
                converted_date = convert_arabic_date(date_info.get('date'))
                if converted_date:
                    if re.match(r'\d{4}-\d{2}-\d{2}', converted_date):
                        safe_add(g, date_uri, LEGAL.date, Literal(converted_date, datatype=XSD.date))
                    else:
                        safe_add(g, date_uri, LEGAL.date, Literal(converted_date))
                if 'event' in date_info:
                    event_entity = create_entity(g, 'Event', date_info['event'])
                    safe_add(g, date_uri, LEGAL.event, event_entity)

    # Handle key issues
    if 'key_issues' in data:
        for issue in data['key_issues']:
            issue_entity = create_entity(g, 'KeyIssue', issue)
            safe_add(g, case, LEGAL.hasKeyIssue, issue_entity)

    # Handle arguments presented in the case
    if 'arguments_presented' in data:
        arguments = URIRef(case + "_arguments")
        safe_add(g, case, LEGAL.hasArgumentsPresented, arguments)
        safe_add(g, arguments, RDF.type, LEGAL.ArgumentsPresented)
        safe_add(g, arguments, RDFS.label, Literal(f"Arguments Presented in Case {case_number}"))  # Add label
        for key, value in data['arguments_presented'].items():
            argument_entity = create_entity(g, sanitize_property(key), value)
            safe_add(g, arguments, getattr(LEGAL, sanitize_property(key)), argument_entity)

    # Handle court's findings
    if 'courts_findings' in data:
        findings = URIRef(case + "_findings")
        safe_add(g, case, LEGAL.hasCourtFindings, findings)
        safe_add(g, findings, RDF.type, LEGAL.CourtFindings)
        safe_add(g, findings, RDFS.label, Literal(f"Court Findings for Case {case_number}"))  # Add label
        for key, value in data['courts_findings'].items():
            if key == 'legal_principles_applied':
                for principle in value:
                    principle_entity = create_entity(g, 'LegalPrinciple', principle)
                    safe_add(g, findings, LEGAL.hasLegalPrincipleApplied, principle_entity)
            else:
                finding_entity = create_entity(g, sanitize_property(key), value)
                safe_add(g, findings, getattr(LEGAL, sanitize_property(key)), finding_entity)

    # Handle case outcome
    if 'outcome' in data:
        outcome = URIRef(case + "_outcome")
        safe_add(g, case, LEGAL.hasOutcome, outcome)
        safe_add(g, outcome, RDF.type, LEGAL.Outcome)
        safe_add(g, outcome, RDFS.label, Literal(f"Outcome of Case {case_number}"))  # Add label
        for key, value in data['outcome'].items():
            outcome_entity = create_entity(g, sanitize_property(key), value)
            safe_add(g, outcome, getattr(LEGAL, sanitize_property(key)), outcome_entity)

    # Handle additional notes
    if 'additional_notes' in data:
        notes = URIRef(case + "_notes")
        safe_add(g, case, LEGAL.hasAdditionalNotes, notes)
        safe_add(g, notes, RDF.type, LEGAL.AdditionalNotes)
        safe_add(g, notes, RDFS.label, Literal(f"Additional Notes for Case {case_number}"))  # Add label
        for key, value in data['additional_notes'].items():
            note_entity = create_entity(g, sanitize_property(key), value)
            safe_add(g, notes, getattr(LEGAL, sanitize_property(key)), note_entity)

    return g

def process_folder(folder_path, output_file, ontology_file):
    """Processes a folder of JSON files and converts each to RDF format."""
    ontology_graph = Graph()
    ontology_graph.parse(ontology_file, format='turtle')
    combined_graph = Graph()

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                graph = json_to_rdf(file_path, ontology_graph)
                combined_graph += graph
                print(f"Successfully processed {filename}")
            except Exception as e:
                logging.error(f"Error processing file {filename}: {str(e)}")
                print(f"Error processing file {filename}. See log for details.")

    # Serialize the combined graph to Turtle format
    combined_graph.serialize(destination=output_file, format='turtle')
    print(f"Conversion complete. Results written to {output_file}")
    print(f"Error log written to conversion_errors.log")

def main():
    parser = argparse.ArgumentParser(description="Convert JSON legal cases to RDF triples.")
    parser.add_argument('--folder', type=str, required=True, help="Path to the folder containing JSON files.")
    parser.add_argument('--output', type=str, required=True, help="Path for the output Turtle file.")
    parser.add_argument('--ontology', type=str, required=True, help="Path to the ontology file.")
    parser.add_argument('--log', type=str, default='conversion_errors.log', help="Path to the log file.")

    args = parser.parse_args()

    # Set up logging configuration using the log file path from arguments
    logging.basicConfig(
        filename=args.log, 
        level=logging.ERROR,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    # Call the process folder function with the provided arguments
    process_folder(args.folder, args.output, args.ontology)

if __name__ == "__main__":
    main()
