"""tables_creation

Revision ID: 28a65604f39d
Revises: 
Create Date: 2020-11-18 09:45:12.947689

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '28a65604f39d'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('horses',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('unibet_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('horse_race', sa.String(), nullable=True),
    sa.Column('father_id', sa.Integer(), nullable=True),
    sa.Column('mother_id', sa.Integer(), nullable=True),
    sa.Column('father_mother_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['father_id'], ['horses.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['father_mother_id'], ['horses.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['mother_id'], ['horses.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_horses_father_id'), 'horses', ['father_id'], unique=False)
    op.create_index(op.f('ix_horses_father_mother_id'), 'horses', ['father_mother_id'], unique=False)
    op.create_index(op.f('ix_horses_horse_race'), 'horses', ['horse_race'], unique=False)
    op.create_index(op.f('ix_horses_mother_id'), 'horses', ['mother_id'], unique=False)
    op.create_index(op.f('ix_horses_name'), 'horses', ['name'], unique=False)
    op.create_index(op.f('ix_horses_unibet_id'), 'horses', ['unibet_id'], unique=True)
    op.create_table('jockeys',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_jockeys_name'), 'jockeys', ['name'], unique=False)
    op.create_table('owners',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_owners_name'), 'owners', ['name'], unique=False)
    op.create_table('race_tracks',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('race_track_name', sa.String(), nullable=False),
    sa.Column('country_name', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_race_tracks_country_name'), 'race_tracks', ['country_name'], unique=False)
    op.create_index(op.f('ix_race_tracks_race_track_name'), 'race_tracks', ['race_track_name'], unique=True)
    op.create_table('stables',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_stables_name'), 'stables', ['name'], unique=False)
    op.create_table('trainers',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_trainers_name'), 'trainers', ['name'], unique=False)
    op.create_table('horse_shows',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('unibet_id', sa.Integer(), nullable=False),
    sa.Column('datetime', sa.DateTime(), nullable=False),
    sa.Column('unibet_n', sa.Integer(), nullable=False),
    sa.Column('ground', sa.String(), nullable=True),
    sa.Column('race_track_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['race_track_id'], ['race_tracks.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_horse_shows_datetime'), 'horse_shows', ['datetime'], unique=False)
    op.create_index(op.f('ix_horse_shows_race_track_id'), 'horse_shows', ['race_track_id'], unique=False)
    op.create_index(op.f('ix_horse_shows_unibet_id'), 'horse_shows', ['unibet_id'], unique=True)
    op.create_index(op.f('ix_horse_shows_unibet_n'), 'horse_shows', ['unibet_n'], unique=False)
    op.create_table('races',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('unibet_id', sa.Integer(), nullable=False),
    sa.Column('unibet_meeting_id', sa.Integer(), nullable=False),
    sa.Column('start_at', sa.DateTime(), nullable=False),
    sa.Column('date', sa.Date(), nullable=False),
    sa.Column('unibet_n', sa.Integer(), nullable=False),
    sa.Column('type', sa.String(), nullable=True),
    sa.Column('conditions', sa.String(), nullable=True),
    sa.Column('stake', sa.Integer(), nullable=True),
    sa.Column('arjel_level', sa.String(), nullable=True),
    sa.Column('distance', sa.Integer(), nullable=False),
    sa.Column('friendly_URL', sa.String(), nullable=True),
    sa.Column('pronostic', sa.String(), nullable=True),
    sa.Column('horse_show_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['horse_show_id'], ['horse_shows.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_races_date'), 'races', ['date'], unique=False)
    op.create_index(op.f('ix_races_horse_show_id'), 'races', ['horse_show_id'], unique=False)
    op.create_index(op.f('ix_races_unibet_id'), 'races', ['unibet_id'], unique=True)
    op.create_index(op.f('ix_races_unibet_meeting_id'), 'races', ['unibet_meeting_id'], unique=True)
    op.create_index(op.f('ix_races_unibet_n'), 'races', ['unibet_n'], unique=False)
    op.create_index('race_code_index', 'races', ['date', 'unibet_n'], unique=False)
    op.create_table('runners',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('unibet_id', sa.Integer(), nullable=False),
    sa.Column('race_id', sa.Integer(), nullable=False),
    sa.Column('jockey_weight', sa.Float(), nullable=True),
    sa.Column('unibet_n', sa.Integer(), nullable=False),
    sa.Column('team', sa.Integer(), nullable=True),
    sa.Column('draw', sa.Integer(), nullable=True),
    sa.Column('blinkers', sa.String(), nullable=True),
    sa.Column('shoes', sa.String(), nullable=True),
    sa.Column('silk', sa.String(), nullable=True),
    sa.Column('bet_counter', sa.Integer(), nullable=True),
    sa.Column('stakes', sa.Integer(), nullable=True),
    sa.Column('music', sa.String(), nullable=True),
    sa.Column('sex', sa.String(), nullable=True),
    sa.Column('age', sa.Integer(), nullable=True),
    sa.Column('coat', sa.String(), nullable=True),
    sa.Column('origins', sa.String(), nullable=True),
    sa.Column('comment', sa.String(), nullable=True),
    sa.Column('owner_id', sa.Integer(), nullable=False),
    sa.Column('trainer_id', sa.Integer(), nullable=False),
    sa.Column('position', sa.String(), nullable=True),
    sa.Column('race_duration_sec', sa.Integer(), nullable=True),
    sa.Column('length', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['owner_id'], ['owners.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['race_id'], ['races.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['trainer_id'], ['trainers.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_runners_age'), 'runners', ['age'], unique=False)
    op.create_index(op.f('ix_runners_bet_counter'), 'runners', ['bet_counter'], unique=False)
    op.create_index(op.f('ix_runners_blinkers'), 'runners', ['blinkers'], unique=False)
    op.create_index(op.f('ix_runners_coat'), 'runners', ['coat'], unique=False)
    op.create_index(op.f('ix_runners_comment'), 'runners', ['comment'], unique=False)
    op.create_index(op.f('ix_runners_draw'), 'runners', ['draw'], unique=False)
    op.create_index(op.f('ix_runners_jockey_weight'), 'runners', ['jockey_weight'], unique=False)
    op.create_index(op.f('ix_runners_music'), 'runners', ['music'], unique=False)
    op.create_index(op.f('ix_runners_origins'), 'runners', ['origins'], unique=False)
    op.create_index(op.f('ix_runners_owner_id'), 'runners', ['owner_id'], unique=False)
    op.create_index(op.f('ix_runners_position'), 'runners', ['position'], unique=False)
    op.create_index(op.f('ix_runners_race_id'), 'runners', ['race_id'], unique=False)
    op.create_index(op.f('ix_runners_sex'), 'runners', ['sex'], unique=False)
    op.create_index(op.f('ix_runners_shoes'), 'runners', ['shoes'], unique=False)
    op.create_index(op.f('ix_runners_silk'), 'runners', ['silk'], unique=False)
    op.create_index(op.f('ix_runners_stakes'), 'runners', ['stakes'], unique=False)
    op.create_index(op.f('ix_runners_team'), 'runners', ['team'], unique=False)
    op.create_index(op.f('ix_runners_trainer_id'), 'runners', ['trainer_id'], unique=False)
    op.create_index(op.f('ix_runners_unibet_id'), 'runners', ['unibet_id'], unique=True)
    op.create_index(op.f('ix_runners_unibet_n'), 'runners', ['unibet_n'], unique=False)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_runners_unibet_n'), table_name='runners')
    op.drop_index(op.f('ix_runners_unibet_id'), table_name='runners')
    op.drop_index(op.f('ix_runners_trainer_id'), table_name='runners')
    op.drop_index(op.f('ix_runners_team'), table_name='runners')
    op.drop_index(op.f('ix_runners_stakes'), table_name='runners')
    op.drop_index(op.f('ix_runners_silk'), table_name='runners')
    op.drop_index(op.f('ix_runners_shoes'), table_name='runners')
    op.drop_index(op.f('ix_runners_sex'), table_name='runners')
    op.drop_index(op.f('ix_runners_race_id'), table_name='runners')
    op.drop_index(op.f('ix_runners_position'), table_name='runners')
    op.drop_index(op.f('ix_runners_owner_id'), table_name='runners')
    op.drop_index(op.f('ix_runners_origins'), table_name='runners')
    op.drop_index(op.f('ix_runners_music'), table_name='runners')
    op.drop_index(op.f('ix_runners_jockey_weight'), table_name='runners')
    op.drop_index(op.f('ix_runners_draw'), table_name='runners')
    op.drop_index(op.f('ix_runners_comment'), table_name='runners')
    op.drop_index(op.f('ix_runners_coat'), table_name='runners')
    op.drop_index(op.f('ix_runners_blinkers'), table_name='runners')
    op.drop_index(op.f('ix_runners_bet_counter'), table_name='runners')
    op.drop_index(op.f('ix_runners_age'), table_name='runners')
    op.drop_table('runners')
    op.drop_index('race_code_index', table_name='races')
    op.drop_index(op.f('ix_races_unibet_n'), table_name='races')
    op.drop_index(op.f('ix_races_unibet_meeting_id'), table_name='races')
    op.drop_index(op.f('ix_races_unibet_id'), table_name='races')
    op.drop_index(op.f('ix_races_horse_show_id'), table_name='races')
    op.drop_index(op.f('ix_races_date'), table_name='races')
    op.drop_table('races')
    op.drop_index(op.f('ix_horse_shows_unibet_n'), table_name='horse_shows')
    op.drop_index(op.f('ix_horse_shows_unibet_id'), table_name='horse_shows')
    op.drop_index(op.f('ix_horse_shows_race_track_id'), table_name='horse_shows')
    op.drop_index(op.f('ix_horse_shows_datetime'), table_name='horse_shows')
    op.drop_table('horse_shows')
    op.drop_index(op.f('ix_trainers_name'), table_name='trainers')
    op.drop_table('trainers')
    op.drop_index(op.f('ix_stables_name'), table_name='stables')
    op.drop_table('stables')
    op.drop_index(op.f('ix_race_tracks_race_track_name'), table_name='race_tracks')
    op.drop_index(op.f('ix_race_tracks_country_name'), table_name='race_tracks')
    op.drop_table('race_tracks')
    op.drop_index(op.f('ix_owners_name'), table_name='owners')
    op.drop_table('owners')
    op.drop_index(op.f('ix_jockeys_name'), table_name='jockeys')
    op.drop_table('jockeys')
    op.drop_index(op.f('ix_horses_unibet_id'), table_name='horses')
    op.drop_index(op.f('ix_horses_name'), table_name='horses')
    op.drop_index(op.f('ix_horses_mother_id'), table_name='horses')
    op.drop_index(op.f('ix_horses_horse_race'), table_name='horses')
    op.drop_index(op.f('ix_horses_father_mother_id'), table_name='horses')
    op.drop_index(op.f('ix_horses_father_id'), table_name='horses')
    op.drop_table('horses')
    # ### end Alembic commands ###
