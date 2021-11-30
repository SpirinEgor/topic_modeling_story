import json
import string
from dataclasses import dataclass
from typing import List, Dict, Tuple, no_type_check

from torch.utils.data import Dataset


@dataclass
class Issue:
    id_readable: str
    created: int
    summary: str
    description: str
    affected_versions: List[str]

    @staticmethod
    @no_type_check
    def from_json(data: Dict) -> "Issue":
        return Issue(
            data.get("idReadable", "") or "",
            data.get("created", 0) or 0,
            data.get("summary", "") or "",
            data.get("description", "") or "",
            data.get("Affected versions", []) or [],
        )


class YouTrackIssueDataset(Dataset):
    _replace_symbols = string.punctuation + "\n\t" + string.digits
    _translator = str.maketrans(_replace_symbols, " " * len(_replace_symbols))

    def __init__(self, input_data_path: str):
        self._issues = []
        with open(input_data_path, "r") as input_file:
            for line in input_file:
                self._issues.append(Issue.from_json(json.loads(line)))

    def __len__(self) -> int:
        return len(self._issues)

    def __getitem__(self, idx: int) -> Tuple[str, List[str]]:
        issue = self._issues[idx]
        return issue.summary, issue.affected_versions
