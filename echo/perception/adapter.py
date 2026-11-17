"""Explicit bridge for the missing parent ObservationPacket schema; caller owns acceptance."""

import copy


def legacy_packet(result, packet_factory, *, extractor):
    data = copy.deepcopy(result["observation"])
    # Prototype Media adds index; original Media has just these five fields.
    fields = {"media_id", "uri", "pts", "time_base", "sha256"}
    data["media"] = [{k: v for k, v in m.items() if k in fields} for m in data["media"]]
    data["extractor"] = extractor
    data["family_id"] = data["observation_id"]
    # Pass ObservationPacket.from_dict as factory. No runtime imports of missing parent code.
    packet = packet_factory(data)
    return packet, copy.deepcopy(result["link_proposals"]), copy.deepcopy(result["coverage_gaps"])
