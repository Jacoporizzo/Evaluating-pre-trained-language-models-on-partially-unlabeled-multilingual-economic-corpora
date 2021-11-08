'''
Topic: Master thesis
Description: Class for importing and preparing data

Created on: 22 October 2021 by Jacopo Rizzo
Last modified on: 22 October 2021
'''
import pickle
import pandas as pd

class Import:
    
    def import8k(self):
        '''
        Function that imports the 8k-filings data.

        Returns
        -------
        data_8k : Dataframe
            Pandas Dataframe containing the 8k-filings data.
            
        '''
        data_8k = pickle.load(open('data/8kDisclosures.pkl', 'rb'))
        return data_8k
    
    def importadhoc(self):
        '''
        Function that imports the ad-hoc announcements.

        Returns
        -------
        data_adhoc : Dataframe
            Pandas Dataframe containing the ad-hoc announcements data.

        '''
        data_adhoc = pickle.load(open('data/adHocNews.pkl', 'rb'))
        return data_adhoc
    
    def importgold(self):
        '''
        Function that imports the Goldstandards data.

        Returns
        -------
        data_goldstandard : Dataframe
            Pandas Dataframe containing the goldstandards data.

        '''
        data_goldstandard = pickle.load(open('data/goldStandardGerman.pkl', 'rb'))
        return data_goldstandard
    
    def findcounterpart(self):
        '''
        Function that returns a merged dataframe between the german labeled 
        announcements and their respective english version.

        Returns
        -------
        english_label : Dataframe
            Merged pandas Dataframe that needs to be reshaped before continuing.

        '''
        # Import data
        data_adhoc = self.importadhoc()
        data_gold = self.importgold()
        
        # Split according to language
        english = data_adhoc[data_adhoc['language'] == 'English'].reset_index(drop = True)
        german = data_adhoc[data_adhoc['language'] == 'Deutsch'].reset_index(drop = True)
        
        # Select form german_adhoc only those documents that have been labeled,
        # i.e. are present in data_goldstandard
        gold_hashs = data_gold['Hashs'].unique()
        german_golds = german[german['hash'].isin(gold_hashs)].reset_index(drop = True)

        english_forlabel = english.merge(german_golds, how = 'inner', on = ['dateText', 'timeText'])
        return english_forlabel