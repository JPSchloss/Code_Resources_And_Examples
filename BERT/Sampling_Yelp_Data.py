import json
import random
import csv

class YelpReviewSampler:
    def __init__(self, input_file_path, output_file_path, sample_size):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.sample_size = sample_size
        self.total_rows = self._count_total_rows()

    def _count_total_rows(self):
        with open(self.input_file_path, "r") as f:
            return sum(1 for _ in f)

    def _calculate_skip_probability(self):
        return 1 - self.sample_size / self.total_rows

    def sample_reviews(self):
        skip_prob = self._calculate_skip_probability()
        sampled_data = []

        with open(self.input_file_path, "r") as f:
            for line in f:
                if random.random() >= skip_prob:
                    data = json.loads(line)
                    sampled_data.append(data)

        self._save_sampled_data(sampled_data)

    def _save_sampled_data(self, sampled_data):
        with open(self.output_file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            
            # Writing the header
            writer.writerow(['review_id', 'user_id', 'business_id', 'stars', 'date', 'text', 'useful', 'funny', 'cool'])
            
            # Writing the rows
            for item in sampled_data:
                writer.writerow([item.get('review_id', ''), item.get('user_id', ''), item.get('business_id', ''),
                                 item.get('stars', ''), item.get('date', ''), item.get('text', '').replace('\n', ' '),
                                 item.get('useful', 0), item.get('funny', 0), item.get('cool', 0)])

if __name__ == "__main__":
    # Input Data Can Be Sourced From Here: https://www.yelp.com/dataset
    input_file_path = "yelp_academic_dataset_review.json"
    output_file_path = "sampled_yelp_reviews.csv"
    sample_size = 200000

    sampler = YelpReviewSampler(input_file_path, output_file_path, sample_size)
    sampler.sample_reviews()

    print(f"Sampled {sample_size} rows from the Yelp dataset.")
