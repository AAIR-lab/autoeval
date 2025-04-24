import pathlib
import json
import argparse

def analyze_file(output_dir, model_name,
                 filter_field,
                 flattened_fh, stats_fh):

    dataset_filepath = "%s/dataset_nlfs_%s_verified.json" % (
        output_dir, model_name)
    with open(dataset_filepath, "r") as fh:
        dataset = json.load(fh)

    if filter_field is None:
        filter_field = dataset["info"]["filter_field"]

    filter_field_stats = {}
    
    for idx, datum in enumerate(dataset["data"]):

        filter_value = datum[filter_field]
        has_error = datum["verification"]["has_error"] is not None
        is_equivalent = datum["verification"]["is_equivalent"]
        if is_equivalent is None:
            is_equivalent = False

        err_list, equiv_list = filter_field_stats.setdefault(filter_value, ([], []))
        err_list.append(int(has_error))
        equiv_list.append(int(is_equivalent))

        flattened_fh.write("%d,%s,%s,%s,%s\n" % (idx,
                                               model_name,
                                               filter_value,
                                               has_error,
                                               is_equivalent))



    for filter_value in filter_field_stats:

        err_list, equiv_list = filter_field_stats[filter_value]
        stats_fh.write("%d,%s,%d,%.2f,%.2f\n" % (filter_value,
                                               model_name,
                                               len(err_list),
                                               sum(err_list) / len(err_list),
                                               sum(equiv_list) / len(equiv_list)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None, type=str,
        required=True)
    parser.add_argument("--filter-field", default=None, type=str)
    parser.add_argument("--models", nargs="+", type=str,
        default=["gpt-3.5-turbo", "gpt-4o"])

    parser.add_argument("--clean", default=False, action="store_true")

    args = parser.parse_args()

    if args.clean:
        mode = "w"
    else:
        mode = "a"

    flattened_fh = open("%s/dataset.flattened.csv" % (args.results_dir), mode)
    stats_fh = open("%s/dataset.stats.csv" % (args.results_dir), mode)
    stats_fh.write("filter_value,model_name,error_total,error_rate,accuracy\n")


    if args.clean:
        flattened_fh.write("idx,model_name,%s,has_error,is_equiv\n" % (args.filter_field))
        stats_fh.write("%s,model_name,total,err_rate,accuracy\n" % (args.filter_field))

    for model_name in args.models:
        analyze_file(args.results_dir, model_name, args.filter_field,
                     flattened_fh,stats_fh)
