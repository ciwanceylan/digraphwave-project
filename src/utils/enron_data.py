import os

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import re

import src.utils.alignment as alignment


def email2index(edges_df: pd.DataFrame, roles_df: pd.DataFrame):
    nodes = pd.concat((edges_df.loc[:, 'source'], edges_df.loc[:, 'target']), ignore_index=True)
    index2email = nodes.drop_duplicates().reset_index(drop=True)
    email2index = pd.Series(dict((v, k) for k, v in index2email.iteritems()))
    email2index.name = "source_index"
    edges_df = edges_df.join(email2index, on="source", how="left")
    email2index.name = "target_index"
    edges_df = edges_df.join(email2index, on="target", how="left")

    edges_df = edges_df.loc[:, ["source_index", "target_index", "count"]]
    edges_df = edges_df.rename(columns={"source_index": "source", "target_index": "target", "count": "count"})

    email2index.name = "node_index"
    roles_df = roles_df.join(email2index, on="email", how="left")
    roles_df = roles_df.loc[:, ["node_index", "role", "additional_info"]]
    return edges_df, roles_df, index2email


def edges2df(sources, targets):
    df = pd.DataFrame({"source": sources, "target": targets})
    invalid_source_email = ~df['source'].str.match(r"[^@]+@[^@]+\.[^@]+")
    invalid_target_email = ~df['target'].str.match(r"[^@]+@[^@]+\.[^@]+")
    df = df.loc[~(invalid_source_email | invalid_target_email)]
    df = df.groupby(["source", "target"]).size().reset_index(name='count')
    return df


def get_roles_df(emplyees_file):
    df = pd.read_csv(emplyees_file)
    df["email"] = df['email_prefix'] + "@enron.com"
    return df


def parse_all_xml_to_edges(rootdir: str):
    all_msg_id = dict()
    all_sources = []
    all_targets = []
    for subdir, dirs, files in tqdm(os.walk(rootdir), total=3500):
        for file in files:
            with open(os.path.join(subdir, file), 'r', encoding='latin-1') as fp:
                try:
                    msg_id, date_line, source, targets = parse_xml_to_edge(fp)
                except AssertionError as e:
                    print(e)
                    print(os.path.join(subdir, file))
                    continue
                except UnicodeDecodeError as e:
                    print(e)
                    print(os.path.join(subdir, file))
                    continue

            if msg_id in all_msg_id:
                print("duplicate!")
                print(os.path.join(subdir, file))
            all_msg_id[msg_id] = os.path.join(subdir, file)
            source = len(targets) * [source]
            all_sources.extend(source)
            all_targets.extend(targets)
    return all_sources, all_targets


def parse_xml_to_edge(fp):
    msg_id = parse_msg_id(fp.readline())
    date_line = parse_date(fp.readline())
    source = parse_from(fp.readline())
    targets = []
    target_triggers = {"To:", "Cc:", "Bcc:"}
    for i in range(20):
        line, match_trigger = _maybe_accumulate_lines(fp, target_triggers)

        if match_trigger:
            for t in target_triggers:
                if line.startswith(t):
                    line = parse_field_line(line, t)
            targets.extend([s.strip('\'\" \t\n\r,') for s in re.split(r"[;,<> ]", line) if "@" in s])
    targets = list(set(targets))
    return msg_id, date_line, source, targets


def _maybe_accumulate_lines(fp, triggers):
    line = fp.readline()
    pos = fp.tell()
    match_trigger = False

    if any(line.startswith(t) for t in triggers):
        match_trigger = True
        next_line = fp.readline()
        while next_line.startswith("\t"):
            line += next_line
            pos = fp.tell()
            next_line = fp.readline()
        fp.seek(pos)
    return line, match_trigger


def parse_field_line(line: str, field: str):
    info = line[len(field):].strip(' \t\n\r,')
    return info


def parse_msg_id(line: str):
    assert line[:len("Message-ID:")] == "Message-ID:"
    return parse_field_line(line, "Message-ID:")


def parse_date(line: str):
    assert line[:len("Date:")] == "Date:"
    return parse_field_line(line, "Date:")


def parse_from(line: str):
    assert line[:len("From:")] == "From:"
    source = parse_field_line(line, "From:")
    assert "," not in source
    assert ";" not in source
    split_sources = [s for s in re.split(r"[<>]", source) if "@" in s]
    source = split_sources[0] if len(split_sources) == 1 else source
    # sources = [s.strip(' \t\n\r,') for s in line.split(",")]
    return source.strip('\'\" \t\n\r,')


# def parse_to(line: str):
#     if not line[:len("To:")] == "To:":
#         return []  # TODO Likely a deleted email or bug, but maybe significant
#     else:
#         line = parse_field_line(line, "To:")
#         targets = [s.strip(' \t\n\r,') for s in line.split(",")]
#     return targets
#
#
# def parse_bcc(line: str):
#     line = parse_field_line(line, "Bcc:")
#     targets = [s.strip(' \t\n\r,') for s in line.split(",")]
#     return targets

##############################

def get_email_domains(enron_emails):
    split_email = enron_emails['email'].str.strip("<>").str.split("@", expand=True, n=1)
    email_domains = split_email[1].astype("category")
    email_domains.name = "email_domain"
    email_domains.index.name = None
    return email_domains


def create_labels():
    folder = "./data/enron/parsed_enron/"
    emails = pd.read_csv(os.path.join(folder, "enron_index2email.csv"), names=["index", "email"], index_col="index")
    split_emails = emails['email'].str.rsplit('@', 1, expand=True).rename(columns={0: "email_name", 1: "email_domain"})

    is_bot = is_bot_heuristic(split_emails)

    is_enron = split_emails['email_domain'].isin({"enron.com", "enron.net", "ect.enron.com", "ei.enron.com"}) & ~is_bot
    is_private = ((split_emails['email_domain'].isin({"aol.com", "hotmail.com", "yahoo.com"}) |
                   split_emails['email_domain'].str.contains("msn.com")
                   ) & ~is_bot)
    is_uni = split_emails['email_domain'].str.endswith(".edu") & ~is_bot
    is_gov = split_emails['email_domain'].str.endswith(".gov") & ~is_bot

    _check_mutually_exclusive(is_bot, is_enron, is_private, is_uni, is_gov)
    emails['email_type_label'] = None
    emails.loc[is_bot, 'email_type_label'] = "bot"
    emails.loc[is_enron, 'email_type_label'] = "enron"
    emails.loc[is_private, 'email_type_label'] = "private"
    emails.loc[is_uni, 'email_type_label'] = 'university'
    emails.loc[is_gov, 'email_type_label'] = 'government'

    email_labels = emails.loc[emails['email_type_label'] != None, 'email_type_label']
    email_labels.to_json(os.path.join(folder, "email_type_labels.json"), indent=2)

    roles = pd.read_csv(os.path.join(folder, "enron_roles.csv"), names=["index", "role", "additional_info"], header=0,
                        index_col="index")
    roles = roles.loc[~roles["role"].isnull(), "role"]
    roles.to_json(os.path.join(folder, "role_labels.json"), indent=2)


def create_simplified_labels():
    folder = "./data/enron/parsed_enron/"
    emails = pd.read_csv(os.path.join(folder, "enron_index2email.csv"), names=["index", "email"], index_col="index")
    split_emails = emails['email'].str.rsplit('@', 1, expand=True).rename(columns={0: "email_name", 1: "email_domain"})

    is_bot = is_bot_heuristic(split_emails)

    is_enron = split_emails['email_domain'].isin({"enron.com", "enron.net", "ect.enron.com", "ei.enron.com"}) & ~is_bot
    is_external = ~is_enron & ~is_bot

    _check_mutually_exclusive(is_bot, is_enron, is_external)
    emails['email_type_label'] = None
    emails.loc[is_bot, 'email_type_label'] = "bot"
    emails.loc[is_enron, 'email_type_label'] = "enron"
    emails.loc[is_external, 'email_type_label'] = "external"

    email_labels = emails.loc[emails['email_type_label'] != None, 'email_type_label']
    email_labels.to_json(os.path.join(folder, "simplified_email_type_labels.json"), indent=2)

    def relabel_roles(label):
        if label in {"Employee", "In House Lawyer", "Trader"}:
            return "Employee"
        elif label in {"Vice President", "Director", "Managing Director", "President", "CEO"}:
            return "Upper-management"
        else:
            return label

    roles = pd.read_csv(os.path.join(folder, "enron_roles.csv"), names=["index", "role", "additional_info"], header=0,
                        index_col="index")
    roles["simplifed_role"] = roles['role'].apply(relabel_roles)
    roles = roles.loc[~roles['simplifed_role'].isnull(), "simplifed_role"]
    roles.to_json(os.path.join(folder, "simplified_role_labels.json"), indent=2)


def create_undirected():
    folder = "data/enron/parsed_enron"
    filename = os.path.join(folder, "enron_edges.tsv")
    num_nodes = None
    edges_df = pd.read_csv(filename, sep="\s+", index_col=False, header=None, comment="%")

    with open(filename) as f:
        first_line = f.readline()
    if first_line[0] in "#%":
        try:
            num_nodes = int(first_line.strip(f"{first_line[0]}\n "))
        except ValueError:
            pass
    if num_nodes is None:
        num_nodes = int(np.max(edges_df.loc[:, [0, 1]])) + 1

    combined = pd.concat((edges_df.loc[:, [0, 1, 2]], edges_df.loc[:, [1, 0, 2]]), ignore_index=True)
    edges_df = combined.groupby([0, 1], as_index=False).agg('mean')

    save_edges(os.path.join(folder, "enron_edges_undirected.tsv"), edges_df, num_nodes)


def _check_mutually_exclusive(*args):
    without_intersect = np.logical_or.reduce(args).sum()
    with_intersect = np.sum([val.sum() for val in args])
    assert without_intersect == with_intersect


def is_bot_heuristic(data: pd.DataFrame):
    email_name_len = data['email_name'].str.len()
    email_domain_len = data['email_domain'].str.len()
    email_name_num_letters = data['email_name'].str.count(r'[a-zÀ-ÿ]')
    num_3_consec_letters = data['email_name'].str.count(r'[a-zÀ-ÿ.]{3}')
    random_mix_letters_and_numbers = (
            (num_3_consec_letters < (0.75 * (email_name_len / 3)).round()) &
            (email_name_num_letters < 0.9 * email_name_len)
    )
    name_is_mosly_numbers = (email_name_num_letters < 0.1 * email_name_len)
    newletter_domain = data['email_domain'].str.contains("newletter")
    postmaster_domain = data['email_domain'].str.contains("postmaster")
    xgate_domain = data['email_domain'].str.lower().str.contains("enronxgate")
    is_maybe_bot = (
            random_mix_letters_and_numbers |
            name_is_mosly_numbers |
            newletter_domain |
            postmaster_domain |
            xgate_domain
    )
    return is_maybe_bot


def save_edges(path, edges_df, num_nodes):
    with open(path, 'w') as fp:
        fp.write("%" + str(num_nodes) + "\n")
        edges_df.to_csv(fp, sep="\t", index=False, header=False)


def parse_dataset():
    email_dir = "./data/enron/maildir/"
    employees_file = "./data/enron/enron_employees.csv"
    assert os.path.exists(email_dir)
    assert os.path.exists(employees_file)

    folder = "./data/enron/parsed_enron/"
    os.makedirs(folder, exist_ok=True)

    sources, targets = parse_all_xml_to_edges(email_dir)
    edges_df = edges2df(sources, targets)
    roles_df = get_roles_df(employees_file)
    edges_df, roles_df, index2email = email2index(edges_df, roles_df)

    save_edges(os.path.join(folder, "enron_edges.tsv"), edges_df, len(index2email))
    roles_df.to_csv(os.path.join(folder, "enron_roles.csv"), index=False)
    index2email.to_csv(os.path.join(folder, "enron_index2email.csv"), index=True, header=False)


def load_labels(data_dir: str, label_type: str):
    if label_type == "email_type":
        email_type_labels = pd.read_json(os.path.join(data_dir, "email_type_labels.json"), typ="series")
        labels = email_type_labels.loc[~email_type_labels.isnull()]
    elif label_type == "role":
        role_labels = pd.read_json(os.path.join(data_dir, "role_labels.json"), typ="series")
        labels = role_labels.loc[~role_labels.isnull()]
    else:
        raise ValueError(f"Unknown label type {label_type}")
    return labels, labels.unique()


def load_simplified_labels(data_dir: str, label_type: str):
    # if data_dir is None:
    #     data_dir = f"data/enron/parsed_enron"

    if label_type == "email_type":
        email_type_labels = pd.read_json(os.path.join(data_dir, "simplified_email_type_labels.json"), typ="series")
        labels = email_type_labels.loc[~email_type_labels.isnull()]
    elif label_type == "role":
        role_labels = pd.read_json(os.path.join(data_dir, "simplified_role_labels.json"), typ="series")
        labels = role_labels.loc[~role_labels.isnull()]
    else:
        raise ValueError(f"Unknown label type {label_type}")
    return labels, labels.unique()


def create_internal_external_email_subgraph(undirected: bool = False):
    undirected_str = "_undirected" if undirected else ""
    folder = "data/enron/parsed_enron"

    folder_internal = os.path.join(folder, "internal_subgraph")
    folder_external = os.path.join(folder, "external_subgraph")
    os.makedirs(folder_internal, exist_ok=True)
    os.makedirs(folder_external, exist_ok=True)

    filename = os.path.join(folder, f"enron_edges{undirected_str}.tsv")
    edges_df = pd.read_csv(filename, sep="\s+", index_col=False, header=None, comment="%")
    emails = pd.read_csv(os.path.join(folder, "enron_index2email.csv"), names=["index", "email"], index_col="index")
    split_emails = emails['email'].str.rsplit('@', 1, expand=True).rename(columns={0: "email_name", 1: "email_domain"})
    is_enron = split_emails['email_domain'].str.contains("enron")

    enron_node_indices = set(emails.index[is_enron])
    internal_edges_mask = edges_df[0].isin(enron_node_indices) & edges_df[1].isin(enron_node_indices)
    internal_edges = pd.DataFrame(edges_df.loc[internal_edges_mask, :])

    if undirected:
        internal2all = alignment.AlignedGraphs.load_from_file(
            os.path.join(folder_internal, f"node_index_alignment.csv")).g2_to_g1
    else:
        internal2all = pd.Series(pd.concat((internal_edges[0], internal_edges[1]), ignore_index=True).unique())
        with open(os.path.join(folder_internal, f"node_index_alignment.csv"), 'w') as fp:
            alignment.AlignedGraphs(emails.shape[0], len(internal2all), internal2all).save2file(fp)

    all2internal = pd.Series(internal2all.index, index=internal2all)
    internal_edges["reindexed_source"] = internal_edges[0].map(all2internal)
    internal_edges["reindexed_target"] = internal_edges[1].map(all2internal)

    save_edges(os.path.join(folder_internal, f"enron_edges{undirected_str}.tsv"),
               internal_edges.loc[:, ["reindexed_source", "reindexed_target", 2]],
               len(internal2all))

    if not undirected:
        roles = pd.read_json(os.path.join(folder, "simplified_role_labels.json"), typ="series")
        internal_index_roles = pd.Series(roles.values, index=roles.index.map(all2internal))
        internal_index_roles.to_json(os.path.join(folder_internal, "simplified_role_labels.json"), indent=2)

    external_edges = pd.DataFrame(edges_df.loc[~internal_edges_mask, :])
    if undirected:
        exterenal2all = alignment.AlignedGraphs.load_from_file(
            os.path.join(folder_external, f"node_index_alignment.csv")).g2_to_g1
    else:
        exterenal2all = pd.Series(pd.concat((external_edges[0], external_edges[1]), ignore_index=True).unique())
        with open(os.path.join(folder_external, f"node_index_alignment.csv"), 'w') as fp:
            alignment.AlignedGraphs(emails.shape[0], len(exterenal2all), exterenal2all).save2file(fp)
    all2external = pd.Series(exterenal2all.index, index=exterenal2all)
    external_edges["reindexed_source"] = external_edges[0].map(all2external)
    external_edges["reindexed_target"] = external_edges[1].map(all2external)

    save_edges(os.path.join(folder_external, f"enron_edges{undirected_str}.tsv"),
               external_edges.loc[:, ["reindexed_source", "reindexed_target", 2]],
               len(exterenal2all))


#
# def get_unique_labels(data_dir=None):
#     if data_dir is None:
#         data_dir = f"data/enron/parsed_enron"
#     email_type_labels = pd.read_json(os.path.join(data_dir, "email_type_labels.json"), typ="series")
#
#     role_labels = pd.read_json(os.path.join(data_dir, "role_labels.json"), typ="series")
#
#     return {"email_type": email_type_labels.unique(), "role": role_labels.unique()}


if __name__ == "__main__":
    # parse_dataset()
    # create_labels()
    # create_simplified_labels()
    # create_undirected()
    create_internal_external_email_subgraph(False)
    create_internal_external_email_subgraph(True)
