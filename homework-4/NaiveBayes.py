import math

class NaiveBayesClassifier():
    def __init__(self):
        self.prior = dict()
        self.conditional = dict()
        self.log_posterior = dict()
        self.log_posteriore = dict()
        self.log_conditional = dict()
        self.log_conditionale = dict()
        self.chars = []
        self.languages = []

    def train(self, data, characters, languages, alpha, lang_length, letter_length):
        self.chars = characters
        self.languages = languages

        for language in languages:
            char_dict = dict()
            for char in characters:
                current_chars = 0 
                total_chars = 0 
                for i in range(0,len(data)):
                    if data[i]['files'] == language:
                        total_chars += data[i]['numchars']
                        for j in range(0,data[i]['numchars']):
                            if data[i]['char'][j] == char:
                                current_chars += 1
                char_dict[char] = (current_chars+alpha)/(total_chars+(letter_length*alpha))
            self.conditional[language] = char_dict
            
        for language in languages:
            total = 0
            for i in range(0,len(data)):
                if data[i]['files'] == language:
                    total += 1
            self.prior[language] = (total+alpha)/(len(data)+(lang_length*alpha))

    def predict(self, sample):
        for language in self.languages:
            log_conditional = 0
            for char in self.chars:
                temp=(sample[char] * math.log10(self.conditional[language][char]))
                log_conditional =log_conditional + temp
            self.log_conditional[language] = log_conditional
            
        for language in self.languages:
            log_conditionale = 0
            for char in self.chars:
                temp= (sample[char] * math.log(self.conditional[language][char]))
                log_conditionale =log_conditionale + temp
            self.log_conditionale[language] = log_conditionale
            
        max_conditional=-5000
        max_conditionale=-10000
        for language in self.languages:
            if self.log_conditional[language]>max_conditional:
                max_conditional=self.log_conditional[language]
                max_conditionale=self.log_conditionale[language]

        for language in self.languages:
            self.log_posterior[language] = self.log_conditional[language]+math.log10(self.prior[language])-max_conditional
            
        for language in self.languages:
            self.log_posteriore[language] = self.log_conditionale[language] + math.log(self.prior[language])-max_conditionale
        
        return max(self.log_posterior, key=self.log_posterior.get)

    def get_prior(self):
        return self.prior

    def get_conditional(self):
        return self.conditional

    def get_log_conditional(self):
        return self.log_conditional
    
    def get_log_conditionale(self):
        return self.log_conditionale

    def get_log_posterior(self):
        return self.log_posterior
    
    def get_log_posteriore(self):
        return self.log_posteriore
