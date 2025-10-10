from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math, random, copy

from .codon_utils import AA_TO_CODONS


class HostConditionalCodonPolicy:
    """
    A simple host-conditional codon policy with per-AA categorical logits.
    - Parameters are stored as: params[host][amino_acid][codon] = logit
    - Sampling uses softmax(logits / temperature)
    - Reference parameters can be used to regularize updates (KL proxy via L2 on logits)
    This is a lightweight scaffold to support GRPO-style training.
    """

    def __init__(self, hosts: List[str], init_usage: Dict[str, float]):
        self.hosts = hosts
        self.params: Dict[str, Dict[str, Dict[str, float]]]= {}
        # initialize logits from usage frequencies (shared across hosts by default)
        eps = 1e-6
        for h in hosts:
            self.params[h] = {}
            for aa, codons in AA_TO_CODONS.items():
                if aa == "*":
                    continue
                logits = {}
                for c in codons:
                    p = max(eps, init_usage.get(c, eps))
                    logits[c] = math.log(p)
                self.params[h][aa] = logits

    def clone(self) -> "HostConditionalCodonPolicy":
        cp = HostConditionalCodonPolicy(self.hosts, {})
        cp.params = copy.deepcopy(self.params)
        return cp

    def _softmax_probs(self, logits: Dict[str, float], temperature: float) -> List[Tuple[str, float]]:
        z = [ (c, l/ max(1e-6, temperature)) for c,l in logits.items() ]
        m = max(l for _,l in z)
        exps = [ (c, math.exp(l - m)) for c,l in z ]
        s = sum(v for _,v in exps) or 1.0
        return [ (c, v/s) for c,v in exps ]

    def sample_codon(self, aa: str, host: str, temperature: float = 1.0) -> Tuple[str, float]:
        logits = self.params[host][aa]
        probs = self._softmax_probs(logits, temperature)
        r = random.random(); cum=0.0
        for c,p in probs:
            cum += p
            if r <= cum:
                return c, math.log(max(1e-12, p))
        c, p = probs[-1]
        return c, math.log(max(1e-12, p))

    def sample_sequence(self, aa_seq: str, host: str, motifs_forbidden: Optional[List[str]] = None, temperature: float = 1.0, max_attempts_per_pos: int = 5) -> Tuple[str, float]:
        dna_chunks: List[str] = []
        logp_sum = 0.0
        for i, aa in enumerate(aa_seq.strip().upper()):
            attempts = 0
            chosen = None
            chosen_logp = 0.0
            while attempts < max_attempts_per_pos:
                attempts += 1
                if i == 0 and aa == "M":
                    cand = "ATG"; logp = 0.0  # treat as fixed start codon
                else:
                    cand, logp = self.sample_codon(aa, host, temperature=temperature)
                trial = "".join(dna_chunks) + cand
                if not motifs_forbidden:
                    chosen, chosen_logp = cand, logp; break
                bad = False
                mmtrial = trial.upper().replace("U","T")
                for m in motifs_forbidden:
                    mm = m.upper().replace("U","T")
                    if mm in mmtrial:
                        bad = True; break
                if not bad:
                    chosen, chosen_logp = cand, logp; break
            if chosen is None:
                chosen, chosen_logp = self.sample_codon(aa, host, temperature=temperature)
            dna_chunks.append(chosen)
            logp_sum += chosen_logp
        return "".join(dna_chunks), logp_sum

    def update_from_samples(
        self,
        host: str,
        aa_seqs: List[str],
        chosen_codons: List[List[str]],
        advantages: List[float],
        ref_policy: Optional["HostConditionalCodonPolicy"] = None,
        lr: float = 0.1,
        beta_ref: float = 0.01,
    ) -> None:
        """
        Apply a simple per-token logit update:
          logit[c] += lr * (adv - beta_ref*(logit[c] - ref_logit[c])) for chosen token c
        This acts like policy gradient with an L2 pull to reference logits (KL proxy).
        aa_seqs and chosen_codons are aligned per sample.
        """
        for aa_seq, codons, adv in zip(aa_seqs, chosen_codons, advantages):
            for aa, c in zip(aa_seq.strip().upper(), codons):
                if aa == "*":
                    continue
                if aa not in self.params[host]:
                    continue
                ref_logit = 0.0
                if ref_policy is not None:
                    ref_logit = ref_policy.params[host].get(aa, {}).get(c, 0.0)
                self.params[host][aa][c] = self.params[host][aa].get(c, 0.0) + lr * (adv - beta_ref * (self.params[host][aa].get(c, 0.0) - ref_logit))


