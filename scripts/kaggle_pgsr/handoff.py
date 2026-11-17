"""Hash-sealed PGSR source and output transport, reusing Phase02 safe bundle helpers."""
from pathlib import Path
import argparse
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts/kaggle"))
from workflow import seal, sha256, read_json, write_json, inside

INPUT_ARCHIVE_SHA = "1de01f37d36adb4a035f988375dc0e24f032fed7e0f7ba227cccadbca3cd24eb"
INPUT_MANIFEST_SHA = "17b05d77ba69da5811ed81f319ed430325c3b73dfcd8deecc98bc8d9703732e3"
SCHEMAS = ("phase03-pgsr-source-v1","phase03-pgsr-output-v1","phase03-pgsr-evaluation-v1")


def verify(folder):
    # Reuse path/hash inventory validation without changing B0's accepted schemas.
    manifest = read_json(folder/"manifest.json")
    if manifest.get("schema") not in SCHEMAS:
        raise ValueError("Expected PGSR source/output schema")
    for rel,digest in manifest["files"].items():
        path = inside(folder,rel)
        if not path.is_file() or sha256(path)!=digest:
            raise ValueError("PGSR bundle checksum mismatch: "+rel)
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if actual != set(manifest["files"])|{"manifest.json"}:
        raise ValueError("Unlisted PGSR bundle contents")
    return manifest


def package(destination):
    from src.pgsr.native import code_identity
    destination.mkdir(parents=True,exist_ok=False)
    capsule = destination/"capsule"
    capsule.mkdir()
    with zipfile.ZipFile(capsule/"source.zip","x",zipfile.ZIP_DEFLATED) as archive:
        for directory in ("src","configs/training","scripts/kaggle","scripts/kaggle_pgsr","scripts/pgsr"):
            for path in sorted((ROOT/directory).rglob("*")):
                if path.is_file() and path.suffix in (".py",".yaml",".json",".sh") and "__pycache__" not in path.parts:
                    archive.write(path,path.relative_to(ROOT).as_posix())
    manifest = seal(capsule,{"schema":"phase03-pgsr-source-v1","source_code_hash":code_identity(),
                    "prepared_archive_sha256":INPUT_ARCHIVE_SHA,"prepared_manifest_sha256":INPUT_MANIFEST_SHA,
                    "prepared_dataset":"vthkhnhhuyn/ignatius-phase02-3dgs-resume-input-v1",
                    "initialization":"Only prepared scene/ from existing sealed input; NO B0 checkpoint"})
    bundle = destination/"phase03_pgsr_source.bundle"
    with zipfile.ZipFile(bundle,"x",zipfile.ZIP_DEFLATED) as archive:
        for path in capsule.rglob("*"):
            if path.is_file():
                archive.write(path,path.relative_to(destination).as_posix())
    # Upload only transport files, not an automatically expanded nested source tree.
    write_json(destination/"bundle_transport.json",{"schema":"phase03-pgsr-transport-v1",
               "archive":bundle.name,"archive_sha256":sha256(bundle),
               "manifest_sha256":sha256(capsule/"manifest.json"),"source_code_hash":manifest["source_code_hash"]})
    return read_json(destination/"bundle_transport.json")


def extract_scene(archive,destination):
    if sha256(archive)!=INPUT_ARCHIVE_SHA:
        raise ValueError("Existing Phase02 prepared input archive hash mismatch")
    destination.mkdir(parents=True,exist_ok=False)
    prefix = "ignatius_phase02_input_v1/"
    with zipfile.ZipFile(archive) as bundle:
        raw = bundle.read(prefix+"manifest.json")
        import hashlib,json
        if hashlib.sha256(raw).hexdigest()!=INPUT_MANIFEST_SHA:
            raise ValueError("Prepared source manifest mismatch")
        manifest = json.loads(raw)
        for name,digest in manifest["files"].items():
            if not name.startswith("scene/"):
                continue
            target = inside(destination,name.removeprefix("scene/"))
            info = bundle.getinfo(prefix+name)
            if (info.external_attr>>16)&0o170000==0o120000:
                raise ValueError("Symlink in sealed scene")
            target.parent.mkdir(parents=True,exist_ok=True)
            with bundle.open(info) as source,target.open("xb") as out:
                shutil.copyfileobj(source,out,1024*1024)
            if sha256(target)!=digest:
                raise ValueError("Prepared scene file mismatch: "+name)


def export(run,output,source_manifest):
    from src.pgsr.checkpoint import inspect
    if output.exists():
        raise FileExistsError(output)
    pointer = read_json(run/"checkpoints/latest.json")
    checkpoint = inside(run/"checkpoints",pointer["path"])
    if sha256(checkpoint)!=pointer["sha256"]:
        raise ValueError("PGSR checkpoint checksum mismatch")
    validation = inspect(checkpoint,read_json(run/"identity.json"))
    files = [p for p in run.rglob("*") if p.is_file() and
             (p.suffix!=".pth" or p==checkpoint) and p.suffix!=".tmp"]
    needed = sum(p.stat().st_size for p in files)
    output.parent.mkdir(parents=True,exist_ok=True)
    used = sum(p.stat().st_size for p in output.parent.rglob("*") if p.is_file())
    if needed+used>18*1024**3 or needed*1.1>shutil.disk_usage(output.parent).free:
        raise ValueError("PGSR export exceeds available space /18GiB safety budget")
    for path in files:
        target = output/"run"/path.relative_to(run)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    complete = (run/"evaluation/result.json").is_file() and validation["iteration"]==read_json(run/"resolved_config.json")["iterations"]
    return seal(output,{"schema":"phase03-pgsr-output-v1","method":"B2_PGSR",
                "run_name":run.name,"iteration":validation["iteration"],"complete":complete,
                "source_manifest_sha256":source_manifest,"checkpoint_validation":validation})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--package",type=Path,required=True)
    print(package(parser.parse_args().package))
