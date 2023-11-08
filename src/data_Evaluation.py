from sdv.evaluation.single_table import evaluate_quality, get_column_plot
from sdv.metadata import SingleTableMetadata
from table_evaluator import TableEvaluator
import json
import os
from src.logging import logger
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Evaluation_Data:
    def __init__(self, real, synthetic):
        self.real_data = real
        self.synthetic_data = synthetic
        self.ax=None
        self.root_folder_path = os.path.join('results', 'evaluation')
        self.fname = 'results/evaluation/'
        os.makedirs(self.root_folder_path, exist_ok=True)

    
    def convert_numerical(self):
        real = self.real_data.select_dtypes(include=[np.number])
        fake = self.synthetic_data.select_dtypes(include=[np.number])
        return real, fake
    

    def plot_mean_std(self):
        
        if self.ax is None:
            fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle('Absolute Log Mean and STDs of numeric data\n', fontsize=16)

        self.ax[0].grid(True)
        self.ax[1].grid(True)
        self.real_data = self.real_data.select_dtypes(include='number')
        self.synthetic_data = self.synthetic_data.select_dtypes(include='number')
        real_mean = np.log(np.add(abs(self.real_data.mean()).values, 1e-5))
        fake_mean = np.log(np.add(abs(self.synthetic_data.mean()).values, 1e-5))
        min_mean = min(real_mean) - 1
        max_mean = max(real_mean) + 1
        line = np.arange(min_mean, max_mean)
        sns.lineplot(x=line, y=line, ax=self.ax[0])
        sns.scatterplot(x=real_mean,
                        y=fake_mean,
                        ax=self.ax[0])
        self.ax[0].set_title('Means of real and fake data')
        self.ax[0].set_xlabel('real data mean (log)')
        self.ax[0].set_ylabel('fake data mean (log)')

        real_std = np.log(np.add(self.real_data.std().values, 1e-5))
        fake_std = np.log(np.add(self.synthetic_data.std().values, 1e-5))
        min_std = min(real_std) - 1
        max_std = max(real_std) + 1
        line = np.arange(min_std, max_std)
        sns.lineplot(x=line, y=line, ax=self.ax[1])
        sns.scatterplot(x=real_std,
                        y=fake_std,
                        ax=self.ax[1])
        self.ax[1].set_title('Stds of real and fake data')
        self.ax[1].set_xlabel('real data std (log)')
        self.ax[1].set_ylabel('fake data std (log)')

        if self.fname is not None:
            plt.savefig(self.fname + 'mean_std.png')
        # plt.show()


    def cumsum_plot(self, nr_cols=4):

        nr_charts = len(self.real_data.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        if not self.real_data.select_dtypes(include=['object']).empty:
            lengths = []
            for d in self.real_data.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in self.real_data[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Cumulative Sums per feature', fontsize=16)
        axes = ax.flatten()

        for i, col in enumerate(self.real_data.columns):
            try:
                r = self.real_data[col]
                f = self.synthetic_data.iloc[:, self.real_data.columns.get_loc(col)]
                x1 = r.sort_values()
                x2 = f.sort_values()
                y = np.arange(1, len(r) + 1) / len(r)

                axis_font = {'size': '14'}
                axes[i].set_xlabel(col, **axis_font)
                axes[i].set_ylabel('Cumulative Sum', **axis_font)

                axes[i].grid()
                axes[i].plot(x1, y, marker='o', linestyle='none', label='Real', ms=8)
                axes[i].plot(x2, y, marker='o', linestyle='none', label='Fake', alpha=0.5)
                axes[i].tick_params(axis='both', which='major', labelsize=8)
                axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

                if isinstance(r, pd.Series) and r.dtypes == 'object':
                    all_labels = set(r) | set(f)
                    ticks_loc = axes[i].get_xticks()
                    axes[i].xaxis.set_major_locator(plt.FixedLocator(ticks_loc))
                    axes[i].set_xticklabels(sorted(all_labels), rotation='vertical')

            except Exception as e:
                print(f'Error while plotting column {col}')
                raise e

        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        if self.fname is not None:
            plt.savefig(self.fname+'cumsums.png')
        # plt.show()


    def plot_pca(self):
        
        real, fake = self.convert_numerical()

        pca_r = PCA(n_components=2)
        pca_f = PCA(n_components=2)

        real_t = pca_r.fit_transform(real)
        fake_t = pca_f.fit_transform(fake)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('First two components of PCA', fontsize=16)
        sns.scatterplot(ax=ax[0], x=real_t[:, 0], y=real_t[:, 1])
        sns.scatterplot(ax=ax[1], x=fake_t[:, 0], y=fake_t[:, 1])
        ax[0].set_title('Real data')
        ax[1].set_title('Fake data')

        if self.fname is not None:
            plt.savefig(self.fname + 'pca.png')
        # plt.show()


    def plot_distributions(self, nr_cols=3):
        
        nr_charts = len(self.real_data.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        real = self.real_data.infer_objects()
        fake = self.synthetic_data.infer_objects()
        self.numerical_columns = [column for column in real.select_dtypes(include='number').columns if
                                  len(real[column].unique()) > 0]
        self.categorical_columns = [column for column in real.columns if column not in self.numerical_columns]
        self.n_samples = min(len(self.real_data), len(self.synthetic_data))

        max_len = 0
        # Increase the length of plots if the labels are long
        if not self.real_data.select_dtypes(include=['object']).empty:
            lengths = []
            for d in self.real_data.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in self.real_data[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Distribution per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(self.real_data.columns):
            if col not in self.categorical_columns:
                plot_df = pd.DataFrame({col: pd.concat([self.real_data[col], self.synthetic_data[col]], axis=0), 'kind': ['real'] * self.n_samples + ['fake'] * self.n_samples})
                fig = sns.histplot(plot_df, x=col, hue='kind', ax=axes[i], stat='probability', legend=True, kde=True)
                axes[i].set_autoscaley_on(True)
            else:
                real = self.real_data.copy()
                fake = self.synthetic_data.copy()
                real['kind'] = 'Real'
                fake['kind'] = 'Fake'
                concat = pd.concat([fake, real])
                palette = sns.color_palette(
                    [(0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
                     (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)])
                x, y, hue = col, "proportion", "kind"
                ax = (concat[x]
                      .groupby(concat[hue])
                      .value_counts(normalize=True)
                      .rename(y)
                      .reset_index()
                      .pipe((sns.barplot, "data"), x=x, y=y, hue=hue, ax=axes[i], saturation=0.8, palette=palette))
                ax.set_xticklabels(axes[i].get_xticklabels(), rotation='vertical')
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        if self.fname is not None:
            plt.savefig(self.fname + 'distributions.png')
        # plt.show()


    def data_scores(self):
        # Creating metadata object for a single table
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=self.real_data)

        quality_report = evaluate_quality(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            metadata=metadata
        )

        print('quality_report\n', quality_report)

        self.overall_quality_report = quality_report.get_score()
        self.properties = quality_report.get_properties()
        self.column_wise_score = quality_report.get_details(property_name='Column Shapes')

        # Convert DataFrames to JSON
        self.properties_json = self.properties.to_dict(orient="records")
        self.column_wise_score_json = self.column_wise_score.to_dict(orient="records")

        # saving SDV plots
        for col in self.real_data.columns:
            fig = get_column_plot(
                real_data=self.real_data,
                synthetic_data=self.synthetic_data,
                column_name=col,
                metadata=metadata
            )
            image_path = os.path.join(self.root_folder_path, f"{col}.png")
            fig.write_image(image_path)


    def json_dump(self):
        data = dict()

        # Define the get_file_paths function
        def get_file_paths(root_folder):
            file_paths = []

            for foldername, subfolders, filenames in os.walk(root_folder):
                for filename in filenames:
                    file_path = os.path.join('results', 'evaluation', filename)
                    # print(file_path)
                    file_paths.append(file_path)
            return file_paths

        # Get a list of image file paths
        if os.path.exists(self.root_folder_path):
            try:
                images_file_paths_list = get_file_paths(self.root_folder_path)
            except Exception as e:
                logger.exception(e)
                images_file_paths_list = []
            if not images_file_paths_list:
                print(f"The specified root folder '{self.root_folder_path}' is empty.")
        else:
            print(f"The specified root folder '{self.root_folder_path}' does not exist.")

        # Create the JSON data structure
        data["Overall Quality Score"] = {
            "one_liner": "Overall Quality Score",
            "Overall Quality Score": self.overall_quality_report
        }

        data["properties"] = {
            "one_liner": "Column lavel overall score",
            "properties": self.properties_json
        }

        data["column_wise_score"] = {
            "one_liner": "Individual Column level score",
            "properties": self.column_wise_score_json
        }

        data["evaluator_images"] = {
            "one_liner": "Table evaluator plots",
            "images_paths": [str(image) for image in images_file_paths_list]
        }

        # Write the data to a JSON file
        with open('results/json_data.json', 'w') as fp:
            json.dump(data, fp)
            logger.info("JSON file is dumped successfully")

    
class Evaluation:
    def __init__(self, real, synthetic):
        self.real_data = real
        self.synthetic_data = synthetic

        if len(self.real_data) != len(self.synthetic_data):
            self.n_samples = min(len(self.real_data), len(self.synthetic_data))

        self.real_data = self.real_data[:self.n_samples]
        self.synthetic_data = self.synthetic_data[:self.n_samples]

    def data_evaluation(self):
        
        # Creating the Evaluatio class object
        obj = Evaluation_Data(self.real_data, self.synthetic_data)
        
        # Call the methods of the Evaluation_Data instance
        obj.plot_mean_std()
        obj.cumsum_plot()
        obj.plot_pca()
        obj.plot_distributions()
        obj.data_scores()
        obj.json_dump()