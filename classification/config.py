import concepts

example_name = {
    'SetFit/sst2': 'text', 
    "fancyzhx/ag_news": 'text', 
    "fancyzhx/yelp_polarity": 'text', 
    "fancyzhx/dbpedia_14": 'content', 
    "TimSchopf/medical_abstracts": 'medical_abstract', 
    "darklord1611/legal_citations": 'case_text', 
    "darklord1611/ecom_categories": 'text', 
    "Duyacquy/Stack_overflow_question": 'Text',
    "Duyacquy/Pubmed_20k": 'abstract_text',
} # Done
concepts_from_labels = {'SetFit/sst2': ["negative","positive"], "fancyzhx/yelp_polarity": ["negative","positive"], "fancyzhx/ag_news": ["World", "Sports", "Business", "Sci/Tech"], "fancyzhx/dbpedia_14": [
        "company",
        "educational institution",
        "artist",
        "athlete",
        "office holder",
        "mean of transportation",
        "building",
        "natural place",
        "village",
        "animal",
        "plant",
        "album",
        "film",
        "written work"
    ], "Duyacquy/Single_label_medical_abstract": ["neoplasms", "digestive_system_diseases", "nervous_system_diseases", "cardiovascular_diseases", "general_pathological_diseases"], "darklord1611/legal_citations": ["affirmed", "applied", "approved", "cited", "considered", "discussed", "distinguished", "followed", "referred to", "related"], "darklord1611/ecom_categories": ["Household", "Books", "Electronics", "Clothing & Accsessories"], 
    "Duyacquy/Stack_overflow_question": ["HQ", "LQ_EDIT", "LQ_CLOSE"],
    "Duyacquy/Pubmed_20k": ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
} # Done

class_num = {'SetFit/sst2': 2, "fancyzhx/ag_news": 4, "fancyzhx/yelp_polarity": 2, "fancyzhx/dbpedia_14": 14, "Duyacquy/Single_label_medical_abstract": 5, "darklord1611/legal_citations": 10, "darklord1611/ecom_categories": 4, "Duyacquy/Stack_overflow_question": 3, "Duyacquy/Pubmed_20k": 5} # Done

# Config for Roberta-Base baseline
finetune_epoch = {'SetFit/sst2': 3, "fancyzhx/ag_news": 2, "fancyzhx/yelp_polarity": 2, "fancyzhx/dbpedia_14": 2}
finetune_mlp_epoch = {'SetFit/sst2': 30, "fancyzhx/ag_news": 5, "fancyzhx/yelp_polarity": 3, "fancyzhx/dbpedia_14": 3}

# Config for CBM training
concept_set = {
    'SetFit/sst2': concepts.sst2, 
    "fancyzhx/yelp_polarity": concepts.yelpp, 
    "fancyzhx/ag_news": concepts.agnews, 
    "fancyzhx/dbpedia_14": concepts.dbpedia, 
    "Duyacquy/Single_label_medical_abstract": concepts.med_abs, 
    "darklord1611/legal_citations": concepts.legal, 
    "darklord1611/ecom_categories": concepts.ecom, 
    "Duyacquy/Stack_overflow_question": concepts.stackoverflow,
    "Duyacquy/Pubmed_20k": concepts.pubmed
}


cbl_epochs = {
    'SetFit/sst2': 10, 
    "fancyzhx/ag_news": 10, 
    "fancyzhx/yelp_polarity": 10, 
    "fancyzhx/dbpedia_14": 10, 
    "Duyacquy/Single_label_medical_abstract": 10, 
    "darklord1611/legal_citations": 10, 
    "darklord1611/ecom_categories": 10, 
    "Duyacquy/Stack_overflow_question": 10,
    "Duyacquy/Pubmed_20k": 10
}

dataset_config = {
    "Duyacquy/Single_label_medical_abstract": {
        "text_column": "medical_abstract",
        "label_column": "condition_label"
    },
    "SetFit/20_newsgroups": {
        "text_column": "text",
        "label_column": "label"
    },
    "JuliaTsk/yahoo-answers": {
        "text_column": "question title",
        "label_column": "class id"
    },
    "fancyzhx/ag_news": {
        "text_column": "text",
        "label_column": "label"
    },
    "fancyzhx/dbpedia_14": {
        "text_column": "content",
        "label_column": "label"
    },
    "SetFit/sst2": {
        "text_column": "text",
        "label_column": "label"
    },
    "fancyzhx/yelp_polarity": {
        "text_column": "text",
        "label_column": "label"
    },
    "pietrolesci/pubmed-200k-rct": {
        "text_column": "text",
        "label_column": "labels"
    },
    "dd-n-kk/uci-drug-review-cleaned": {
        "text_column": "review",
        "label_column": "rating"
    },
    "darklord1611/legal_citations": {
        "text_column": "case_text",
        "label_column": "case_outcome"
    },
    "darklord1611/ecom_categories": {
        "text_column": "text",
        "label_column": "label"
    },
    "Duyacquy/Stack_overflow_question": {
        "text_column": "Text",
        "label_column": "Y"
    },
    "Duyacquy/Pubmed_20k": {
        "text_column": "abstract_text",
        "label_column": "target"
    }
}
